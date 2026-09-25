#!/usr/bin/env python3
import argparse
import sys
import time
import serial

NAMES = {
    1: "shoulder_pan (베이스)",
    2: "shoulder_lift (어깨)",
    3: "elbow_flex (팔꿈치)",
    4: "wrist_flex (손목 상하)",
    5: "wrist_roll (손목 회전)",
    6: "gripper (그리퍼)",
}

def scan_once(port: str, baud: int = 1000000):
    try:
        ser = serial.Serial(port, baud, timeout=0.04)
    except Exception as e:
        print(f"❌ 포트 열기 실패 ({port}): {e}")
        return [], [1, 2, 3, 4, 5, 6]

    found, missing = [], []
    for mid in range(1, 7):
        pkt = bytes([0xFF, 0xFF, mid, 0x02, 0x01, (~(mid + 3)) & 0xFF])
        ser.reset_input_buffer()
        ser.write(pkt)
        time.sleep(0.005)
        res = ser.read(6)
        if len(res) >= 6 and res[0] == 0xFF and res[1] == 0xFF and res[2] == mid:
            found.append(mid)
            print(f"  ID {mid} [{NAMES[mid]}]: ✅ 정상 응답!")
        else:
            missing.append(mid)
            print(f"  ID {mid} [{NAMES[mid]}]: ❌ 무응답 (Missing)")
    ser.close()
    return found, missing

def main():
    parser = argparse.ArgumentParser(description="SO-101 Feetech Motor Bus Scanner")
    parser.add_argument("--port", default="/dev/so101_follower", help="Serial port to scan")
    parser.add_argument("--loop", action="store_true", help="Continuously scan every 1s")
    args = parser.parse_args()

    print("=" * 60)
    print(f"🔍 [SO-101 모터 버스 스캐너] 포트: {args.port}")
    print("=" * 60)

    if args.loop:
        print("반복 감지 모드 실행 중... (케이블을 연결/분리하면서 확인하세요 / 종료: Ctrl+C)\n")
        try:
            while True:
                found, missing = scan_once(args.port)
                status = f"✅ 발견 {len(found)}개 / ❌ 누락 {len(missing)}개: {found}"
                print(f"[{time.strftime('%H:%M:%S')}] {status}\n")
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n스캔 종료.")
    else:
        found, missing = scan_once(args.port)
        print("-" * 60)
        print(f"📊 스캔 결과: 발견된 모터 {found} / 누락된 모터 {missing}")
        print("=" * 60)

if __name__ == "__main__":
    main()
