# RoboMaster Explore & Engage — 8 files

โค้ด V20.8 จาก `Speed editor(2).py` แยกตามระบบเป็น **8 ไฟล์ Python หลัก**
วาง `main.py` ไว้ด้านนอก ส่วนโค้ดอีก 7 ไฟล์อยู่ใน `src/`
มี `src/__init__.py` เป็นไฟล์ประกาศแพ็กเกจ และไม่มีสคริปต์สำรองปนอยู่

## แก้อะไร เปิดไฟล์ไหน

| ไฟล์ | หน้าที่ |
| --- | --- |
| `main.py` | รับ arguments และเปิด GUI / headless mission |
| `src/config.py` | ค่าตั้งทั้งหมด, ขนาดสนาม, origin/start cell และ helper บันทึกไฟล์ |
| `src/robot_control.py` | เชื่อมต่อ SDK, PID, เดิน, หมุน, gimbal, เบรกและกู้คืนการเคลื่อนที่ |
| `src/sensors.py` | Telemetry, callbacks, ToF, Sharp, IR, freshness และ helpers เรขาคณิต |
| `src/navigation.py` | สถานะหุ่น, DFS, แผนที่, BFS/Dijkstra, บันทึก/โหลด snapshot และภารกิจ Round 2 |
| `src/vision.py` | Camera thread, ปรับแสง, ตรวจสี/รูปทรง, ROI และยืนยันเป้าหลายเฟรม |
| `src/targeting.py` | กวาดหาเป้า, เลื่อนมุมมอง, เล็ง, เงื่อนไขยิงและจำเป้า |
| `src/gui.py` | หน้าควบคุมทั้งหมด, speed editor, arena editor และแสดงแผนที่ |

หน้าแรกของโปรเจกต์มี `main.py`, `README.md`, `requirements.txt` และโฟลเดอร์ `src/`
ส่วน `maps/` จะถูกสร้างเมื่อมีการบันทึกผลภารกิจ
ไฟล์ของระบบใหญ่ยังมีหลักพันบรรทัด เพื่อให้ logic ของระบบเดียวกันอยู่ด้วยกัน

## เริ่มใช้งาน

แตก ZIP แล้วเปิด terminal ในโฟลเดอร์ที่มี `main.py`
ใช้ Python environment เดิมที่รัน RoboMaster SDK และกล้องได้:

```bash
python main.py
```

ดู options โดยไม่เชื่อมต่อหุ่น:

```bash
python main.py --help
```

ตัวอย่าง:

```bash
python main.py --round 1 --fire INFRARED --shots 1
python main.py --round 2
python main.py --grid-width 6 --grid-height 6 --start-x 1 --start-y 0
```

ถ้าสร้าง environment ใหม่ ติดตั้ง dependencies ด้วย:

```bash
python -m pip install -r requirements.txt
```

รายการนี้ไม่ใช่ lockfile ที่ยืนยันกับเครื่องแข่งขัน ให้ยึดเวอร์ชันจาก environment
ที่ใช้งานกับหุ่นได้แล้ว Tkinter มากับ Python/ระบบปฏิบัติการ; บน Ubuntu ใช้
`sudo apt install python3-tk` หากไม่มี และ camera preview ต้องใช้ OpenCV ที่รองรับ GUI

ชุด 8 ไฟล์ใช้ `main.py` เป็นทางเข้าแทน `Speed editor.py`
ถ้าจะใช้ผลรอบแรกที่เคยบันทึก ให้นำโฟลเดอร์ `maps/` เดิมมาวางข้าง `main.py`
output ยังคงอ้างอิง working directory ปัจจุบัน จึงควรรันจากโฟลเดอร์โปรเจกต์

**พฤติกรรมเดิมยังคงอยู่:** `--no-gui` เชื่อมต่อหุ่นและเริ่มภารกิจทันที
ไม่ใช่ simulation; ค่าเริ่มต้นเปิด automatic firing แบบ INFRARED / 1 shot
เมื่อ GUI ใช้งานไม่ได้ โปรแกรมยัง fallback ไป headless เหมือนต้นฉบับ

## ปรับค่าตั้ง

แก้ค่าที่ `src/config.py` หรือใช้ speed/arena editor ใน GUI ก่อนเริ่มภารกิจ
โมดูลอื่นอ่านค่าผ่าน module เดียวกัน:

```python
from src import config as cfg

speed = cfg.DFS_EXPLORE_SPEED_MPS
```

อย่าคัดลอกค่าด้วย `from config import DFS_EXPLORE_SPEED_MPS`
เพราะเมื่อ GUI เปลี่ยนค่า ตัวแปรที่คัดลอกมาอาจยังเป็นค่าเดิม
สำหรับสนาม ให้ใช้ `src.config.apply_arena_profile()` เพื่อปรับ bounds และ start cell พร้อมกัน
ค่า default arguments และ derived aliases ยังคำนวณตอน import ตามต้นฉบับ

## การประกอบระบบ

- `src.navigation.DFSMapOnlyExplorer` ใช้ความสามารถจาก `RobotControlMixin` และ `SensorMixin`
  บนสถานะหุ่นชุดเดียวกัน และสร้าง target subsystem ของตัวเอง
- `src.vision.TargetVisionSubsystem` ใช้ `TargetingMixin` สำหรับกวาด เล็งและยิง
  โดยข้อมูลเป้าและ locks ยังอยู่บน instance เดียวกัน
- `src.gui.MissionControlGUI` เรียกใช้งาน explorer ตัวนั้นโดยตรง

## ผลตรวจฉบับนี้

- ตรวจ normalized AST ครบ 248 ฟังก์ชัน/เมธอด เทียบกับต้นฉบับ
- ค่าตั้งต้น 467 ค่าและ API ของคลาสเดิมทั้ง 5 คลาสตรงกัน
- Offline regression tests ผ่าน 11 รายการ: config ข้ามไฟล์, speed editor,
  ขอบเขตสนาม, BFS/Dijkstra, Round-2 order, map/snapshot round trip,
  telemetry freshness, PID และ CLI
- รวมเฉพาะโค้ดรันจริงใน ZIP; ชุดตรวจ offline ไม่ได้รวมมาด้วย
- ยังไม่ได้ตรวจการเคลื่อนที่บนหุ่นจริง การตรวจจับจากกล้องจริง หรือการแสดงผล GUI

ตรวจ syntax โดยไม่เชื่อมต่อหุ่นได้ด้วย:

```bash
python -m compileall -q main.py src
```
