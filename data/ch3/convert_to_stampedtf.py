import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped

def read_poses_and_broadcast(file_path1, file_path2):
    # Initialize ROS node
    rospy.init_node('pose_broadcaster')

    # Create a TransformBroadcaster
    broadcaster = tf2_ros.TransformBroadcaster()

    # Open both files
    with open(file_path1, 'r') as file1, open(file_path2, 'r') as file2:
        lines1 = iter(file1.readlines())
        lines2 = iter(file2.readlines())
        
        try:
            line1 = next(lines1)
            line2 = next(lines2)
        except StopIteration:
            return  # One of the files may be empty

        # Initial reading to start the loop
        data1 = line1.split()
        data2 = line2.split()
        time1 = float(data1[0])
        time2 = float(data2[0])

        # Process each file
        while True:
            try:
                print("time diff: ", abs(time1 - time2))
                # Compare timestamps within the tolerance
                if abs(time1 - time2) <= 0.002:
                    # Process both lines since they are synchronized within the tolerance
                    process_line(data1, broadcaster, "mocap", "marker")
                    process_line(data2, broadcaster, "lidar_base", "lidar")

                    # Move to the next lines in both files
                    line1 = next(lines1)
                    line2 = next(lines2)
                    data1 = line1.split()
                    data2 = line2.split()
                    time1 = float(data1[0])
                    time2 = float(data2[0])
                elif time1 < time2:
                    # Advance file1 to catch up to file2
                    line1 = next(lines1)
                    data1 = line1.split()
                    time1 = float(data1[0])
                else:
                    # Advance file2 to catch up to file1
                    line2 = next(lines2)
                    data2 = line2.split()
                    time2 = float(data2[0])
            except StopIteration:
                # End of one of the files
                break

def process_line(data, broadcaster, parent_frame_id, child_frame_id):
    time, x, y, z, qx, qy, qz, qw = data

    # Create a TransformStamped message
    transform_stamped = TransformStamped()
    transform_stamped.header.stamp = rospy.Time.from_sec(float(time))
    transform_stamped.header.frame_id = parent_frame_id
    transform_stamped.child_frame_id = child_frame_id

    # Set the translation
    transform_stamped.transform.translation.x = float(x)
    transform_stamped.transform.translation.y = float(y)
    transform_stamped.transform.translation.z = float(z)

    # Set the rotation
    transform_stamped.transform.rotation.x = float(qx)
    transform_stamped.transform.rotation.y = float(qy)
    transform_stamped.transform.rotation.z = float(qz)
    transform_stamped.transform.rotation.w = float(qw)

    # Broadcast the transform
    broadcaster.sendTransform(transform_stamped)
    rospy.sleep(0.01)

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: python script.py path_to_poses_file1 path_to_poses_file2")
        sys.exit(1)
    file_path1 = sys.argv[1]
    file_path2 = sys.argv[2]
    try:
        read_poses_and_broadcast(file_path1, file_path2)
    except rospy.ROSInterruptException:
        pass
