## image folder demo
# python demo/demo.py image_folder --config config/nanodet-plus-m_416.yml --model checkpoints/nanodet-plus-m_416-fisheye/model_best/model_best.ckpt \
#                                  --path /data/fod/dataset/ocean-park-medium-object/2022-03-10-15-05-25/2d-raw/cam_0/
## fordedge 
#front
# python demo/demo.py image_folder --config config/nanodet-plus-m_416.yml --model checkpoints/nanodet-plus-m_416-fisheye/model_best/model_best.ckpt \
                                #  --path /data/fod/dataset/20220413/batch-5/2022-04-13-17-41-11_46/2d-raw/cam_0/
#rear
# python demo/demo.py image_folder --config config/nanodet-plus-m_416.yml --model checkpoints/nanodet-plus-m_416-fisheye/model_best/model_best.ckpt \
                                #  --path /data/fod/dataset/20220413/batch-5/2022-04-13-17-41-11_46/2d-raw/cam_3/ 
python demo/demo.py image_folder --config config/nanodet-plus-m_416.yml --model checkpoints/nanodet-plus-m_416-fisheye/model_best/model_best.ckpt \
                                 --path /data/fod/dataset/20220413/batch-5/2022-04-13-17-44-56_51/2d-raw/cam_3/ 
# luxsa
# python demo/demo.py image_folder --config config/nanodet-plus-m_416.yml --model checkpoints/nanodet-plus-m_416-fisheye/model_best/model_best.ckpt \
                                #  --path /data/fod/dataset/self-label/task_bv_thanhnhan_camfront-2022_05_19_10_24_55-coco/images/
#e34
# python demo/demo.py image_folder --config config/nanodet-plus-m_416.yml --model checkpoints/nanodet-plus-m_416-fisheye/model_best/model_best.ckpt \
#                                  --path /data/low-light/sample_dataset/fromQC/basement_217lux_moving/                                
#e34
# python demo/demo.py image_folder --config config/nanodet-plus-m_416-fisheye.yml --model checkpoints/nanodet-plus-m_416-fisheye/model_best/model_best.ckpt \
                                #  --path //data/low-light/sample_dataset/fromQC/Shadow_13000_Moving/ 

######### single image/video demo
# python demo/demo_video.py video --config config/nanodet-plus-m_416.yml --model checkpoints/nanodet-plus-m_416-fisheye/model_best/model_best.ckpt \
                                #  --path /data/360view/data20210603/di_qua_duong_sat/cam2_2.avi --track

# python demo/demo_video.py image --config config/nanodet-plus-m_416.yml --model checkpoints/nanodet-plus-m_416-fisheye/model_best/model_best.ckpt \
#                                  --path /data/fod/fisheye_od/arm64/data/test/test2_1776.jpg --save_result

######### pseudo 
# python demo/pseudo.py image_folder --config config/nanodet-plus-m_416.yml --model checkpoints/nanodet-plus-m_416-fisheye/model_best/model_best.ckpt \
#                                  --path /data/fod/dataset/sample4TeamData/fisheye_images/ --pseudo