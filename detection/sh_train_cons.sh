python train_byol.py --weights weights/yolov5m.pt --cfg models/yolov5m.yaml --data data/Exdark_night.yaml --batch-size 32 \
--epochs 30 --imgsz 608 --name 32_Exdark_night_mask --hyp data/hyps/hyp.scratch-high.yaml \
--back_ratio 0.3 --byol_weight 0.1 --multi_val

python train_byol.py --weights weights/yolov5m.pt --cfg models/yolov5m.yaml --data data/Exdark_night.yaml --batch-size 16 \
--epochs 30 --imgsz 608 --name 16_Exdark_night_mask --hyp data/hyps/hyp.scratch-high.yaml \
--back_ratio 0.3 --byol_weight 0.1 --multi_val

python train_byol.py --weights weights/yolov5m.pt --cfg models/yolov5m.yaml --data data/Exdark_night.yaml --batch-size 8 \
--epochs 30 --imgsz 608 --name 8_Exdark_night_mask --hyp data/hyps/hyp.scratch-high.yaml \
--back_ratio 0.3 --byol_weight 0.1 --multi_val

python train_byol.py --weights weights/yolov5m.pt --cfg models/yolov5m.yaml --data data/Exdark_night.yaml --batch-size 4 \
--epochs 30 --imgsz 608 --name 4_Exdark_night_mask --hyp data/hyps/hyp.scratch-high.yaml \
--back_ratio 0.3 --byol_weight 0.1 --multi_val



