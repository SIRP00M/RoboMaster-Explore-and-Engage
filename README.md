# RoboMaster EP - Autonomous Maze Exploration & Target Engagement

ระบบสำรวจเขาวงกตอัตโนมัติ (Autonomous Maze Navigation) สำหรับหุ่นยนต์ **DJI RoboMaster EP** โดยใช้อัลกอริทึม **Depth-First Search (DFS)** บน Grid Map พร้อมระบบตรวจจับเป้าหมายด้วย **Computer Vision (OpenCV & SDK Markers)**, ยิงเป้าหมายด้วย **Infrared Blaster**, และควบคุมภารกิจผ่าน **Mission Control GUI (Tkinter)**

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```text
RoboMaster-Explore-and-Engage/
├── main.py                          # จุดเริ่มต้นโปรแกรมหลัก (CLI & Mission Launcher)
├── config/                          # ศูนย์รวมค่าคงที่และการตั้งค่าทั้งหมด
│   ├── __init__.py                  # รวมการตั้งค่าทั้งหมด
│   ├── hardware.py                  # พอร์ตเซนเซอร์, Sensor Adapter IDs, ความถี่ Telemetry
│   ├── calibration.py               # Sharp IR ADC Lookup Tables & Filter Settings
│   ├── motion.py                    # ความเร็ว, Yaw PID, Authority Arbitration, Turn Parameters
│   ├── maze.py                      # ขนาด Cell (0.60m), ToF Open Threshold, Start Anchor
│   ├── vision.py                    # ค่าสี HSV, Bounding Box ROI, Foam Wall Horizon Gate
│   └── blaster.py                   # กฎการยิง (Fire Policy), ระยะยิงจำกัด (<= 1.2m)
├── src/                             # แพ็กเกจโมดูลระบบหลัก (Modular Architecture)
│   ├── __init__.py
│   ├── core/                        # โครงสร้างพื้นฐาน
│   │   ├── geometry.py              # ทิศทาง (N, E, S, W), การคำนวณมุม, wrap_deg, clamp
│   │   └── state.py                 # คลาส SharedState (Thread-safe Telemetry Cache)
│   ├── hardware/                    # ฮาร์ดแวร์และการเชื่อมต่อ
│   │   ├── robot_base.py            # HardwareMixin (Robot Lifecycle & Sensor Callbacks)
│   │   ├── sensor_manager.py        # SensorManagerMixin (ToF, Sharp ADC, IR Filter)
│   │   └── sensors.py               # ฟังก์ชันแปลง Sharp ADC -> cm (adc_to_cm)
│   ├── motion/                      # การควบคุมการเคลื่อนที่
│   │   └── motion_controller.py     # MotionMixin (Corridor Centering, Yaw Hold, Turns, Recovery)
│   ├── vision/                      # การประมวลผลภาพ
│   │   ├── vision_pipeline.py       # VisionMixin (OpenCV Stream, Foam Horizon Gate)
│   │   └── target_tracker.py        # TargetTrackerMixin (Target Lock, Glimpse Memory, Fusion)
│   ├── blaster/                     # ระบบยิง Blaster
│   │   └── blaster_controller.py    # BlasterMixin (Auto-fire, Range-gate Verification)
│   ├── mapping/                     # ระบบแผนที่และการนำทาง
│   │   ├── grid_map.py              # GridMapMixin (Maze Graph, Boundary Guard, SVG/ASCII/JSON)
│   │   └── navigator.py             # NavigatorMixin (BFS/Dijkstra, Fast Return Home, Exit Candidate)
│   ├── gui/                         # หน้าต่างควบคุมภารกิจ
│   │   ├── mission_control.py       # คลาส MissionControlGUI (Tkinter Dashboard)
│   │   └── gui_bridge.py            # GuiBridgeMixin (ส่ง Snapshot เข้าคิว GUI)
│   └── strategy/                    # ยุทธศาสตร์ภารกิจ
│       └── explorer.py              # DFSMazeExplorer (Orchestrator รวมทุก Mixins และรัน DFS)
├── tests/                           # เครื่องมือทดสอบและ Calibrate
│   ├── test_sensors.py              # สคริปต์ตรวจเช็กค่าเซนเซอร์สด (Live Diagnostic)
│   ├── calibrate_sharp.py           # เครื่องมือเก็บข้อมูล Calibrate Sharp IR Sensor
│   └── sharp_calibration_runs/      # ข้อมูลดิบและผลการ Calibrate จากการทดสอบจริง
├── maps/                            # โฟลเดอร์เก็บไฟล์แผนที่ JSON / SVG / ASCII อัตโนมัติ
└── archive/                         # กรุเก็บไฟล์ดั้งเดิมและไฟล์สำรองในอดีต
```

---

## 🔌 การต่อวงจรเซนเซอร์ (Hardware Wiring)

| เซนเซอร์ | ตำแหน่ง | การเชื่อมต่อ / ID | รายละเอียด |
| :--- | :--- | :--- | :--- |
| **Digital IR** | ซ้าย (Left) | Sensor Adapter ID 1, Port 1 | Active LOW (0 = ตรวจพบกำแพง, 1 = ทางเปิด) |
| **Digital IR** | ขวา (Right) | Sensor Adapter ID 4, Port 1 | Active LOW (0 = ตรวจพบกำแพง, 1 = ทางเปิด) |
| **Sharp GP2Y0A41SK0F** | ซ้าย (Left) | Sensor Adapter ID 2, Port 1 | Analog Pin (ADC อ่านค่า 4 - 24 cm) |
| **Sharp GP2Y0A41SK0F** | ขวา (Right) | Sensor Adapter ID 3, Port 1 | Analog Pin (ADC อ่านค่า 4 - 24 cm) |
| **ToF Distance Sensor** | กิมบอล (Gimbal) | CAN bus `distance_info[0]` | วัดระยะทางด้านหน้าและมุมสแกนกิมบอล |
| **RoboMaster Camera** | กิมบอล (Gimbal) | Native Camera Stream | วิดีโอ 360p สำหรับ OpenCV ตรวจจับเป้าหมาย |
| **Infrared Blaster** | กิมบอล (Gimbal) | Native Blaster | ยิงลำแสงอินฟราเรดเข้าเป้าหมายที่ระยะ <= 1.2 เมตร |

---

## 📦 การติดตั้งไลบรารี (Installation)

```bash
# ติดตั้งแพ็กเกจ Python ทั้งหมดที่จำเป็น
pip install -r requirements.txt

# สำหรับ Linux / Ubuntu หากต้องการใช้งาน Mission Control GUI (Tkinter)
sudo apt-get install python3-tk
```

---

## 🚀 วิธีการใช้งาน (How to Run)

### 1. ทดสอบเซนเซอร์ก่อนเริ่มภารกิจ
```bash
python3 tests/test_sensors.py
```

### 2. รันการสำรวจเขาวงกต (Explore Mode)
สำรวจเขาวงกตอัตโนมัติด้วย DFS พร้อมเปิดหน้าต่าง Mission Control GUI:
```bash
python3 main.py --mode explore
```

### 3. รันโหมด Replay ตามแผนที่ที่เคยสำรวจ (Known-Map Mode)
```bash
python3 main.py --mode known --map maps/latest_map.json
```
หากต้องการระบุพิกัดเป้าหมายปลายทาง (เช่น พิกัด `x=2, y=4`):
```bash
python3 main.py --mode known --map maps/latest_map.json --goal 2,4
```

### 4. รันแบบ Auto (ใช้แผนที่เดิมถ้ามี หรือสำรวจใหม่ถ้ายังไม่มี)
```bash
python3 main.py --mode auto
```

### 5. รันผ่าน Terminal เท่านั้น (ปิด GUI)
```bash
python3 main.py --mode explore --no-gui
```

---

## ⚙️ การตั้งค่าและจูนพารามิเตอร์ (Configuration)

หากต้องการปรับแต่งพารามิเตอร์ สามารถเข้าไปแก้ไขในโฟลเดอร์ `config/` ได้ทันทีโดยไม่ต้องแก้โค้ดตรรกะ:
- **`config/motion.py`**: ความเร็วเดินหน้า (`FORWARD_SPEED_MPS`), ค่าเกน PID เกาะกำแพง (`SHARP_FOLLOW_KP`), ค่าหมุนเลี้ยว
- **`config/vision.py`**: ช่วงค่าสีเป้าหมาย HSV (`TARGET_HSV_RANGES`), ความสูงขอบโฟม (`TARGET_FOAM_HSV_LOW/HIGH`)
- **`config/blaster.py`**: ระยะยิงสูงสุด (`TARGET_FIRE_MAX_TILES = 2.0` คือ 120 ซม.), ชนิดเป้าหมายที่อนุญาตให้ยิง
- **`config/maze.py`**: ขนาดช่องตาราง (`GRID_TILE_M = 0.60`), ค่า Threshold ผนังเปิด ToF (`TOF_OPEN_THRESHOLD_MM = 600`)
