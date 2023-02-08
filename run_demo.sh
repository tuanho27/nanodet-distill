########### image detection demo ############
# python demo/demo.py image_folder --config config/nanodet-plus-m_416.yml --model checkpoints/nanodet-plus-m_416-fisheye/model_best/model_best.ckpt \
#                                  --path /data/fod/dataset/ocean-park-medium-object/2022-03-10-15-05-25/2d-raw/cam_0/
## fordedge 
#front
# python demo/demo.py image_folder --config config/nanodet-plus-m_416.yml --model checkpoints/nanodet-plus-m_416-fisheye/model_best/model_best.ckpt \
                                #  --path /data/fod/dataset/20220413/batch-5/2022-04-13-17-41-11_46/2d-raw/cam_0/
#rear
# python demo/demo.py image_folder --config config/nanodet-plus-m_416.yml --model checkpoints/nanodet-plus-m_416-fisheye/model_best/model_best.ckpt \
                                #  --path /data/fod/dataset/20220413/batch-5/2022-04-13-17-41-11_46/2d-raw/cam_3/ 
# python demo/demo.py image_folder --config config/nanodet-plus-m_416.yml --model checkpoints/nanodet-plus-m_416-fisheye/model_best/model_best.ckpt \
#                                  --path /data/fod/dataset/20220413/batch-5/2022-04-13-17-44-56_51/2d-raw/cam_3/ 

############## single image/video demo ############
# python demo/demo_video.py video --config config/nanodet-plus-m_416.yml --model checkpoints/nanodet-plus-m_416-fisheye/model_best/model_best.ckpt \
                                #  --path /data/360view/data20210603/di_qua_duong_sat/cam2_2.avi --track

# python demo/demo_video.py image --config config/nanodet-plus-m_416.yml --model checkpoints/nanodet-plus-m_416-fisheye/model_best/model_best.ckpt \
#                                  --path /data/fod/fisheye_od/arm64/data/test/test2_1776.jpg --save_result

############### image nearby threat demo ############
# python demo/demo_nearby_threat.py image_folder --config config/nanodet-EfficientNet-Lite4_320_fisheye-resize.yml \
#                                         --model workspace/efficientlite4_320_fisheye-resize/model_best/model_best.ckpt \
                                        # --path /home/ubuntu/Workspace/tuan-dev/fisheye_od/nanodet/data/1_Friday_noon_outsideVinUni_29seats_whitecar/
                                        # --path /home/ubuntu/Workspace/datasets/od/fisheye_5class/nearby_2/images/
# luxsa
# python demo/demo.py image_folder --config config/nanodet-plus-m_416.yml --model checkpoints/nanodet-plus-m_416-fisheye/model_best/model_best.ckpt \
                                #  --path /data/fod/dataset/self-label/task_bv_thanhnhan_camfront-2022_05_19_10_24_55-coco/images/
#e34
# python demo/demo.py image_folder --config config/nanodet-plus-m_416.yml --model checkpoints/nanodet-plus-m_416-fisheye/model_best/model_best.ckpt \
#                                  --path /data/low-light/sample_dataset/fromQC/basement_217lux_moving/                                
#e34
# python demo/demo.py image_folder --config config/nanodet-plus-m_416-fisheye.yml --model checkpoints/nanodet-plus-m_416-fisheye/model_best/model_best.ckpt \
                                #  --path //data/low-light/sample_dataset/fromQC/Shadow_13000_Moving/ 


############### pseudo  data ################
# python demo/pseudo.py image_folder --config config/nanodet-plus-m_416.yml --model checkpoints/nanodet-plus-m_416-fisheye/model_best/model_best.ckpt \
#                                  --path /data/fod/dataset/sample4TeamData/fisheye_images/ --pseudo


############### QC test data #################
# python demo/demo_nearby_threat.py image_folder --config config/nanodet-plus-m_320_darknet-fisheye.yml \
#                                         --model workspace/darknet_320_fisheye-resize-udpate-regmax-addnearby123motvoc-night-cleandata-5classes/model_best/model_best.ckpt \
#                                         --path data/TEST/

########### Rear AEB demo ################
# python demo/demo_track_raeb.py image_folder --config config/nanodet-plus-m_320_darknet-fisheye.yml \
#                                         --model workspace/darknet_320_fisheye-resize-udpate-regmax-addnearby123motvoc-night-cleandata-5classes/model_best/model_best.ckpt \
#                                         --path /data/fod/dataset/tracking/rear/ego_stopped \
#                                         --save_result


# python demo/demo_track_raeb.py image_folder --config config/nanodet-EfficientNet-Lite4_320_fisheye-resize.yml \
#                                         --model workspace/efficientlite4_320_fisheye-resize-night-cleandata-5classes/model_best/model_best.ckpt \
#                                         --path /data/fod/dataset/tracking/rear/ego_stopped \
#                                         --save_result

# python demo/demo_track_raeb.py video --config config/nanodet-plus-m_320_darknet-fisheye.yml \
#                                         --model workspace/darknet_320_fisheye-resize-udpate-regmax-addnearby123motvoc-night-cleandata-5classes/model_best/model_best.ckpt \
#                                         --path /data/fod/dataset/videos/rear_view_01Feb23/videos/cams_1675244812249.mp4 \
#                                         --save_result

# python demo/demo_track_raeb.py video --config config/nanodet-plus-m_320_darknet-fisheye.yml \
#                                         --model workspace/darknet_320_fisheye-resize-udpate-regmax-addnearby123motvoc-night-cleandata-5classes/model_best/model_best.ckpt \
#                                         --path /data/fod/dataset/videos/rear_view_01Feb23/videos/cams_1675244695984.mp4 \
#                                         --reID_model_path yolov7-deepsort-tracking/deep_sort/model_weights/mars-small128.pb \
#                                         --save_result 

python demo/demo_track_raeb.py video --config config/nanodet-plus-m_320_darknet-fisheye.yml \
                                        --model workspace/darknet_320_fisheye-resize-udpate-regmax-addnearby123motvoc-night-cleandata-5classes/model_best/model_best.ckpt \
                                        --path /data/fod/dataset/videos/rear_view_01Feb23/videos/cams_1675244576348.mp4 \
                                        --reID_model_path yolov7-deepsort-tracking/deep_sort/model_weights/mars-small128.pb \
                                        --save_result                                         