#!/usr/bin/env python3
"""Voice input/output for the rover.

Input:  arecord -> energy VAD -> Groq Whisper (remote).
Output: espeak-ng or Piper (local) -> aplay.

Why remote STT and local TTS: a 2 GB Pi that already thermal-throttles cannot
run Whisper, and you are paying for Groq anyway. TTS the other way round --
short confirmations should not need a network round trip, and espeak-ng is
essentially free CPU-wise.

Devices default to the cards recorded in the project notes:
  mic     card 1 (USB PnP, TI PCM2902)
  speaker card 2 (UACDemoV1.0, stereo -> plughw handles the mono conversion)

Env overrides: ROVER_MIC, ROVER_SPK, ROVER_TTS (espeak|piper),
ROVER_PIPER_MODEL, ROVER_STT_MODEL.
"""

import audioop  # stdlib on Python 3.10; removed in 3.13
import io
import os
import queue
import re
import subprocess
import threading
import time
import wave

MIC = os.environ.get("ROVER_MIC", "plughw:1,0")
SPK = os.environ.get("ROVER_SPK", "plughw:2,0")
TTS_ENGINE = os.environ.get("ROVER_TTS", "espeak")
PIPER_MODEL = os.environ.get("ROVER_PIPER_MODEL", "")
ESPEAK_VOICE = os.environ.get("ROVER_VOICE", "")      # empty = espeak's default
ESPEAK_SPEED = os.environ.get("ROVER_SPEED", "150")
ORPHEUS_MODEL = os.environ.get("ROVER_ORPHEUS_MODEL", "canopylabs/orpheus-v1-english")
ORPHEUS_VOICE = os.environ.get("ROVER_ORPHEUS_VOICE", "hannah")


def _concat_wav(a, b):
    """Join two WAV blobs. Orpheus caps input at 200 chars, so long replies
    come back as several files that must be stitched before playback."""
    out = io.BytesIO()
    with wave.open(io.BytesIO(a)) as wa, wave.open(io.BytesIO(b)) as wb:
        with wave.open(out, "wb") as w:
            w.setparams(wa.getparams())
            w.writeframes(wa.readframes(wa.getnframes()))
            w.writeframes(wb.readframes(wb.getnframes()))
    return out.getvalue()
STT_MODEL = os.environ.get("ROVER_STT_MODEL", "whisper-large-v3-turbo")

RATE = 16000          # what Whisper wants; plughw resamples from the card
CHUNK = 512           # samples -> 32 ms per chunk
BYTES = CHUNK * 2
PREROLL = 10          # chunks of audio kept before speech onset (~320 ms)


class Voice:
    def __init__(self, groq_client, verbose=True):
        self.client = groq_client
        self.verbose = verbose
        self._proc = None
        self._deaf = threading.Event()      # set while the speaker is active
        self._say_q = queue.Queue()
        self._floor = int(os.environ.get("ROVER_MIC_FLOOR", 900))
        self._fixed_floor = "ROVER_MIC_FLOOR" in os.environ
        threading.Thread(target=self._say_worker, daemon=True).start()

    # ------------------------------------------------------------- speaking

    def say(self, text, block=False):
        """Queue text for speech. Thread-safe; safe to call from ROS callbacks."""
        text = (text or "").strip()
        if not text:
            return
        done = threading.Event() if block else None
        self._say_q.put((text, done))
        if done:
            done.wait(timeout=30)

    def _say_worker(self):
        while True:
            text, done = self._say_q.get()
            try:
                wav = self._synth(text)
                if not wav:
                    print("[tts produced no audio -- see the error above]")
                else:
                    self._deaf.set()
                    p = subprocess.run(["aplay", "-q", "-D", SPK, "-"],
                                       input=wav, stderr=subprocess.PIPE)
                    if p.returncode != 0:
                self._flush_mic()       # drain the pipe BEFORE going live
                        print(f"[aplay failed on {SPK}: "
                              f"{p.stderr.decode(errors='replace').strip()}]")
                    time.sleep(0.4)         # let the tail decay before listening
            except FileNotFoundError as e:
                print(f"[tts missing: {e}]")
            except Exception as e:
                print(f"[tts error: {e}]")
            finally:
                self._deaf.clear()
                if done:
                    done.set()

    def _synth(self, text):
        if TTS_ENGINE == "orpheus":
            return self._synth_orpheus(text)
        if TTS_ENGINE == "piper" and PIPER_MODEL:
            cmd = ["piper", "--model", PIPER_MODEL, "--output_file", "-"]
            p = subprocess.run(cmd, input=text.encode(), capture_output=True)
        else:
            # No -v flag by default: the voice name varies between espeak-ng
            # builds, and a bad one exits non-zero with empty stdout. Override
            # with ROVER_VOICE only if you have checked `espeak-ng --voices`.
            cmd = ["espeak-ng", "-s", ESPEAK_SPEED, "--stdout", text]
            if ESPEAK_VOICE:
                cmd[1:1] = ["-v", ESPEAK_VOICE]
            p = subprocess.run(cmd, capture_output=True)

        if p.returncode != 0 or not p.stdout:
            err = p.stderr.decode(errors="replace").strip()
            print(f"[tts: {cmd[0]} exited {p.returncode}, "
                  f"{len(p.stdout)} bytes"
                  + (f' -- "{err[:200]}"' if err else "") + "]")
        return p.stdout

    @staticmethod
    def _flush_mic(self):
        """Discard audio recorded while speaking.

        arecord keeps writing into the pipe during playback, so bytes read
        just after _deaf clears are seconds old -- the rover's own voice.
        """
        p = self._proc
        if p is None or p.poll() is not None:
            return
        fd = p.stdout.fileno()
        os.set_blocking(fd, False)
        n = 0
        try:
            while True:
                c = p.stdout.read(65536)
                if not c:
                    break
                n += len(c)
        except (BlockingIOError, OSError):
            pass
        finally:
            os.set_blocking(fd, True)
            if n:
                print(f"[flushed {n/2/16000:.1f}s of mic audio]")

    def _chunk(text, limit=190):
        """Split into <=limit pieces on sentence boundaries. Orpheus caps at 200."""
        parts, cur = [], ""
        for piece in re.split(r"(?<=[.!?])\s+", text.strip()):
            while len(piece) > limit:                 # pathological single sentence
                cut = piece.rfind(" ", 0, limit)
                cut = cut if cut > 0 else limit
                parts.append(piece[:cut])
                piece = piece[cut:].lstrip()
            if len(cur) + len(piece) + 1 <= limit:
                cur = f"{cur} {piece}".strip()
            else:
                if cur:
                    parts.append(cur)
                cur = piece
        if cur:
            parts.append(cur)
        return parts or [text[:limit]]

    def _synth_orpheus(self, text):
        """Neural TTS on Groq's hardware -- no CPU or heat cost on the Pi."""
        out = b""
        for chunk in self._chunk(text):
            try:
                r = self.client.audio.speech.create(
                    model=ORPHEUS_MODEL, voice=ORPHEUS_VOICE,
                    input=chunk, response_format="wav")
                wav = r.read() if hasattr(r, "read") else r.content
            except Exception as e:
                print(f"[orpheus error: {e}]")
                print("[falling back to espeak for this line]")
                p = subprocess.run(
                    ["espeak-ng", "-s", ESPEAK_SPEED, "--stdout", text],
                    capture_output=True)
                return p.stdout
            out = wav if not out else _concat_wav(out, wav)
        return out

    # ------------------------------------------------------------ listening

    def _stream(self):
        if self._proc is None or self._proc.poll() is not None:
            self._proc = subprocess.Popen(
                ["arecord", "-D", MIC, "-q", "-f", "S16_LE",
                 "-r", str(RATE), "-c", "1", "-t", "raw"],
                stdout=subprocess.PIPE)
        onset = 0
        return self._proc

    def calibrate(self, seconds=1.5, skip=0.4, multiplier=3.0):
        """Measure room noise and set the speech threshold above it.

        The first few hundred ms after arecord starts are a settling transient
        (values several times the true floor), so they are discarded. The
        threshold is based on a high percentile rather than the peak, so one
        stray click cannot pin it above normal speech.
        """
        s = self._stream()
        if self._fixed_floor:
            if self.verbose:
                print(f"[mic threshold fixed at {self._floor} via ROVER_MIC_FLOOR]")
            return self._floor
        for _ in range(int(skip * RATE / CHUNK)):     # drop the transient
            if not s.stdout.read(BYTES):
                break
        levels = []
        for _ in range(int(seconds * RATE / CHUNK)):
            data = s.stdout.read(BYTES)
            if not data:
                break
            levels.append(audioop.rms(data, 2))
        if not levels:
            self._floor = 800
            return self._floor
        levels.sort()
        noise = levels[int(len(levels) * 0.9)]        # 90th percentile
        self._floor = max(250, min(2500, int(noise * multiplier)))
        if self.verbose:
            print(f"[mic calibrated: noise {noise} "
                  f"(min {levels[0]}, max {levels[-1]}), "
                  f"threshold {self._floor}]")
        return self._floor

    def listen(self, silence_ms=1300, max_s=12.0, timeout_s=None):
        """Block until an utterance is captured. Returns raw PCM bytes, or None."""
        s = self._stream()
        pre, frames = [], []
        speaking = False
        quiet = 0
        quiet_limit = int(silence_ms / 1000 * RATE / CHUNK)
        max_chunks = int(max_s * RATE / CHUNK)
        deadline = time.time() + timeout_s if timeout_s else None

        while True:
            data = s.stdout.read(BYTES)
            if not data:
                return None

            # Discard everything captured while we are talking, so the rover
            # does not transcribe its own voice.
            if self._deaf.is_set():
                pre.clear()
                frames.clear()
                speaking = False
                quiet = 0
                continue

            level = audioop.rms(data, 2)

            if not speaking:
                if deadline and time.time() > deadline:
                    return None
                pre.append(data)
                if len(pre) > PREROLL:
                    pre.pop(0)
                if level > self._floor:
                    onset += 1
                    if onset >= 3:          # ~96 ms of sustained energy
                        speaking = True
                        frames = pre[:] + [data]
                        pre = []
                        onset = 0
                else:
                    onset = 0
                continue

            frames.append(data)
            quiet = quiet + 1 if level <= self._floor else 0
            if quiet >= quiet_limit or len(frames) >= max_chunks:
                # Reject blips too short to be a real command.
                if len(frames) < quiet_limit + 8:
                    speaking, frames, quiet, onset = False, [], 0, 0
                    continue
                loud = sum(1 for f in frames if audioop.rms(f, 2) > self._floor)
                if loud / len(frames) < 0.08:
                    if self.verbose:
                        print(f"[discarded: only {loud}/{len(frames)} "
                              f"chunks above floor]")
                    speaking, frames, quiet, onset = False, [], 0, 0
                    continue
                return b"".join(frames)

    def transcribe(self, pcm):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(pcm)
        buf.seek(0)
        try:
            r = self.client.audio.transcriptions.create(
                file=("speech.wav", buf.read()),
                model=STT_MODEL,
                response_format="text",
                language="en")
        except Exception as e:
            print(f"[stt error: {e}]")
            return ""
        text = r if isinstance(r, str) else getattr(r, "text", "")
        return (text or "").strip()

    def listen_once(self, **kw):
        pcm = self.listen(**kw)
        if not pcm:
            return ""
        secs = len(pcm) / 2 / RATE
        if self.verbose:
            print(f"[heard {secs:.1f}s]")
        return self.transcribe(pcm)

    def close(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()


def main():
    """Standalone check: python3 rover_voice.py"""
    from groq import Groq
    v = Voice(Groq(api_key=os.environ["GROQ_API_KEY"]))
    # Calibrate first: powered speakers hiss, and measuring after playback
    # bakes the amplifier's noise floor into the threshold.
    v.calibrate()
    v.say("Microphone test. Say something now.", block=True)
    print("listening...")
    text = v.listen_once()
    print(f"transcript: {text!r}")
    v.say(f"I heard: {text}" if text else "I did not catch that.", block=True)
    v.close()


if __name__ == "__main__":
    main()
