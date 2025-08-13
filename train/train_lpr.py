import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"

from ultralytics import YOLO
import argparse
from utils.paths import BASE_DIR, RUNS_DIR, abspath, DEVICE

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default='config/lpr_data.yaml')
    ap.add_argument('--model', type=str, default='yolov8n.pt')  # 可改 yolov8s.pt
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--project', type=str, default=str(RUNS_DIR / 'train_lpr'))
    ap.add_argument('--name', type=str, default='yolov8_lpr')
    ap.add_argument('--device', type=str, default=DEVICE)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--resume', type=str, default=None)

    # 可选优化项（只保留通用支持的，避免版本报错）
    ap.add_argument('--optimizer', type=str, default='AdamW')
    ap.add_argument('--lr0', type=float, default=0.005)
    ap.add_argument('--lrf', type=float, default=0.01)
    ap.add_argument('--weight_decay', type=float, default=0.0007)
    ap.add_argument('--warmup_epochs', type=float, default=3.0)

    args = ap.parse_args()

    data_yaml = abspath(args.data)
    model_path = abspath(args.model)

    model = YOLO(str(model_path))
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=str(abspath(args.project)),
        name=args.name,
        exist_ok=True,
        device=args.device,
        workers=args.workers,
        resume=args.resume,
        plots=False,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
    )

    print(f"\n[OK] best.pt at {abspath(args.project) / args.name / 'weights' / 'best.pt'}")

if __name__ == '__main__':
    main()
