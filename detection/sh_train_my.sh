python train_cons.py --weights weights/yolov5m.pt --cfg models/yolov5m.yaml --data data/Exdark_night.yaml --batch-size 32 \
--epochs 80 --imgsz 608 --name 32_Exdark_night_my --hyp data/hyps/hyp.scratch-high.yaml --num_crop 3
python train_cons.py --weights weights/yolov5m.pt --cfg models/yolov5m.yaml --data data/Exdark_night.yaml --batch-size 32 \
--epochs 80 --imgsz 608 --name 32_Exdark_night_my --hyp data/hyps/hyp.scratch-high.yaml --num_crop 3 --back_ratio 0.3


python train_cons.py --weights weights/yolov5m.pt --cfg models/yolov5m.yaml --data data/Exdark_night.yaml --batch-size 16 \
--epochs 80 --imgsz 608 --name 16_Exdark_night_my --hyp data/hyps/hyp.scratch-high.yaml --num_crop 3
python train_cons.py --weights weights/yolov5m.pt --cfg models/yolov5m.yaml --data data/Exdark_night.yaml --batch-size 16 \
--epochs 80 --imgsz 608 --name 16_Exdark_night_my --hyp data/hyps/hyp.scratch-high.yaml --num_crop 3 --back_ratio 0.3


python train_cons.py --weights weights/yolov5m.pt --cfg models/yolov5m.yaml --data data/Exdark_night.yaml --batch-size 8 \
--epochs 80 --imgsz 608 --name 8_Exdark_night_my --hyp data/hyps/hyp.scratch-high.yaml --num_crop 3
python train_cons.py --weights weights/yolov5m.pt --cfg models/yolov5m.yaml --data data/Exdark_night.yaml --batch-size 8 \
--epochs 80 --imgsz 608 --name 8_Exdark_night_my --hyp data/hyps/hyp.scratch-high.yaml --num_crop 3 --back_ratio 0.3



python train_cons.py --weights weights/yolov5m.pt --cfg models/yolov5m.yaml --data data/Exdark_night.yaml --batch-size 4 \
--epochs 80 --imgsz 608 --name 4_Exdark_night_my --hyp data/hyps/hyp.scratch-high.yaml --num_crop 3
python train_cons.py --weights weights/yolov5m.pt --cfg models/yolov5m.yaml --data data/Exdark_night.yaml --batch-size 4 \
--epochs 80 --imgsz 608 --name 4_Exdark_night_my --hyp data/hyps/hyp.scratch-high.yaml --num_crop 3 --back_ratio 0.3

