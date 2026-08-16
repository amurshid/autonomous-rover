from setuptools import setup

package_name = 'ugv_imu_bridge'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ahnaf',
    maintainer_email='murshid.ahnaf@gmail.com',
    description='Bridges Waveshare UGV ESP32 UART IMU stream to a ROS2 sensor_msgs/Imu topic.',
    license='GPL-3.0',
    entry_points={
        'console_scripts': [
            'imu_publisher = ugv_imu_bridge.imu_publisher:main',
            'scan_timestamp_relay = ugv_imu_bridge.scan_timestamp_relay:main',
        ],
    },
)
