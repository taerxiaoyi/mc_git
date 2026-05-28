from evdev import InputDevice, categorize, ecodes

dev = InputDevice('/dev/input/event25')  # 改成你的设备

print(dev)

for event in dev.read_loop():
    if event.type == ecodes.EV_KEY:
        print("按钮:", categorize(event))
    elif event.type == ecodes.EV_ABS:
        print("摇杆:", categorize(event))