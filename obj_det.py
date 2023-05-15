import pyzed.sl as sl
import cv2
import numpy as np
import rospy

def main():
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.sdk_verbose = True

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    obj_param = sl.ObjectDetectionParameters()
    obj_param.enable_tracking = True
    obj_param.image_sync = True
    obj_param.enable_mask_output = True

    camera_infos = zed.get_camera_information()
    if obj_param.enable_tracking:
        positional_tracking_param = sl.PositionalTrackingParameters()
        positional_tracking_param.set_floor_as_origin = True
        zed.enable_positional_tracking(positional_tracking_param)

    err = zed.enable_object_detection(obj_param)
    if err != sl.ERROR_CODE.SUCCESS:
        print(repr(err))
        zed.close()
        exit(1)

    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 40

    runtime = sl.RuntimeParameters()
    runtime.sensing_mode = sl.SENSING_MODE.STANDARD

    image_size = zed.get_camera_information().camera_resolution
    image_size.width = image_size.width / 2
    image_size.height = image_size.height / 2
    # 540 * 960

    image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)

    while zed.grab() == sl.ERROR_CODE.SUCCESS:
        err = zed.retrieve_objects(objects, obj_runtime_param)
        if objects.is_new:
            obj_array = objects.object_list
            print("\n" + str(len(obj_array)) + " Object(s) detected\n")
            if len(obj_array) > 0:
                first_object = obj_array[0]
                print("object attributes:")
                print(" Label '" + repr(first_object.label) + "' (conf. " + str(int(first_object.confidence)) + "/100)")

                if obj_param.enable_tracking:
                    print(" Tracking ID: " + str(int(first_object.id)) + " tracking state: " + repr(
                        first_object.tracking_state) + " / " + repr(first_object.action_state))

                bounding_box_2d = first_object.bounding_box_2d
                print(bounding_box_2d)
                bbox = bounding_box_2d

        while True:
            if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
                image_ocv = image_zed.get_data()

                bbox.resize(8, 1)
                pt1 = (int(bbox[0]/2), int(bbox[1]/2))
                pt2 = (int(bbox[4]/2), int(bbox[5]/2))
                pt3 = (int(bbox[2]/2), int(bbox[3]/2))

                cv2.rectangle(image_ocv, pt1, pt2, color=(0, 0, 255), thickness=3)
                cv2.putText(image_ocv, repr(first_object.label), pt1, 0, 1, (0, 0, 255), 2)
                cv2.putText(image_ocv, str(int(first_object.id)), pt3, 0, 1, (0, 0, 255), 2)
                cv2.imshow("Image", image_ocv)
                if cv2.waitKey(1):
                    break

    cv2.destroyAllWindows()
    zed.close()


if __name__ == "__main__":
     main()
