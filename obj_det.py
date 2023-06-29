import pyzed.sl as sl
import cv2, os, sys, time
import numpy as np
import rospy

def main():
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.sdk_verbose = True
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    obj_param = sl.ObjectDetectionParameters()
    obj_param.enable_tracking = True
    obj_param.image_sync = True
    obj_param.enable_mask_output = True
    obj_param.detection_model = sl.DETECTION_MODEL.MULTI_CLASS_BOX

    camera_infos = zed.get_camera_information()
    if obj_param.enable_tracking:
        positional_tracking_param = sl.PositionalTrackingParameters()
        positional_tracking_param.set_floor_as_origin = True
        zed.enable_positional_tracking(positional_tracking_param)
        err = zed.enable_positional_tracking(positional_tracking_param)

    err = zed.enable_object_detection(obj_param)

    if err != sl.ERROR_CODE.SUCCESS:
        print("enable_object_detection", err, "\nExit program.")
        zed.close()
        exit(1)

    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 40

    runtime = sl.RuntimeParameters()
    runtime.sensing_mode = sl.SENSING_MODE.STANDARD

    image_size = zed.get_camera_information().camera_resolution
    image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)

    while zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_objects(objects, obj_runtime_param)
        if objects.is_new:
            obj_array = objects.object_list
            print("\n" + str(len(obj_array)) + " Object(s) detected")
            if len(obj_array) == 0:
                print("객체 감지 재시작\n")
                executable = sys.executable
                args = sys.argv[:]
                args.insert(0, sys.executable)
                time.sleep(1)
                cv2.destroyAllWindows()
                zed.close()
                os.execvp(executable, args)
            else:
                i = 0
                object = obj_array[i]
                for object in objects.object_list:
                    print("{} {} {}".format(object.id, repr(object.label), str(int(object.confidence)) + "/100"))
                    i = i + 1

            zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
            image_ocv = image_zed.get_data()

            while True:
                for j in range(0, i):
                    bbox = obj_array[j].bounding_box_2d
                    bbox.resize(8, 1)
                    pt1 = (int(bbox[0]), int(bbox[1]))
                    pt2 = (int(bbox[4]), int(bbox[5]))
                    pt3 = ((int(bbox[2])) - 120, (int(bbox[3])))
                    pt4 = ((int(bbox[0])), (int(bbox[1])) - 30)

                    cv2.rectangle(image_ocv, pt1, pt2, color=(0, 0, 255), thickness=3)
                    cv2.putText(image_ocv, "label: " + repr(obj_array[j].label), pt4, 0, 1, (0, 0, 255), 2)
                    cv2.putText(image_ocv, "ID: " + str(int(obj_array[j].id)), pt1, 0, 1, (0, 0, 255), 2)
                    cv2.putText(image_ocv, str(int(obj_array[j].confidence)) + "/100", pt3, 0, 1, (0, 0, 255), 2)

                cv2.imshow("Image", image_ocv)
                if cv2.waitKey(1):
                    break

    cv2.destroyAllWindows()
    zed.close()

if __name__ == "__main__":
     main()
