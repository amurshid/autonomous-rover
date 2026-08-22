#!/usr/bin/env python3
"""Voice conversation with the LLM. No ROS, no robot.

Purpose: verify the full speech loop and audition voices before wiring any of
it to the rover. Nothing here touches /cmd_vel or nav2 -- it just talks.

    source ~/.rover_env
    python3 ~/voice_chat.py

Say "goodbye" or press Ctrl-C to exit.
"""

import os
import sys

from groq import Groq

sys.path.insert(0, os.path.expanduser("~"))
from rover_voice import Voice  # noqa: E402

MODEL = os.environ.get("ROVER_LLM_MODEL", "llama-3.3-70b-versatile")
MAX_HISTORY = 20

SYSTEM = (
    "You are the voice of a small four-wheeled robot that drives around a "
    "house. You are chatting, not driving -- you have no controls connected "
    "right now, so if asked to move, say so cheerfully. "
    "Your replies are read aloud by a speech synthesiser, so: one or two "
    "short sentences, plain words, no lists, no markdown, no emoji, no "
    "asterisks. Write numbers as words when short."
)

EXITS = {"goodbye", "good bye", "bye", "exit", "quit", "stop talking"}


def main():
    if "GROQ_API_KEY" not in os.environ:
        print("Set GROQ_API_KEY first (source ~/.rover_env)")
        return

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    voice = Voice(client)
    history = [{"role": "system", "content": SYSTEM}]

    voice.calibrate()
    print(f"Voice chat ready ({MODEL}). Say 'goodbye' to finish.\n")
    voice.say("I am listening.", block=True)

    try:
        while True:
            heard = voice.listen_once()
            if not heard or len(heard) < 2:
                continue
            print(f"you > {heard}")

            if heard.strip().lower().strip(".!?") in EXITS:
                voice.say("Goodbye.", block=True)
                break

            history.append({"role": "user", "content": heard})
            if len(history) > MAX_HISTORY + 1:
                history = [history[0]] + history[-MAX_HISTORY:]

            try:
                r = client.chat.completions.create(
                    model=MODEL, messages=history, max_tokens=150)
                reply = r.choices[0].message.content or "Sorry, I have nothing to say."
            except Exception as e:
                reply = "I could not reach my language model."
                print(f"[llm error: {e}]")

            history.append({"role": "assistant", "content": reply})
            print(f"bot > {reply}\n")
            voice.say(reply, block=True)

    except KeyboardInterrupt:
        print()
    finally:
        voice.close()


if __name__ == "__main__":
    main()
