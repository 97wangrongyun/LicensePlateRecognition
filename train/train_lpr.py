from ultralytics import YOLO
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default='config/lpr_data.yaml')
    ap.add_argument('--model', type=str, default='yolov8n.pt')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--project', type=str, default='runs/train_lpr')
    ap.add_argument('--name', type=str, default='yolov8_lpr')
    ap.add_argument('--device', type=str, default='0', help="GPU编号或'cpu'")
    ap.add_argument('--workers', type=int, default=2, help="数据加载线程数，Windows建议0~2")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        exist_ok=True,
        device=args.device,
        workers=args.workers
    )

    print(f"\n[INFO] Training finished. Check {args.project}/{args.name}/weights/best.pt")

if __name__ == '__main__':
    main()
