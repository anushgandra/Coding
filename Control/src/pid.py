import numpy as np
import rospy
import time

from controller import BaseController


class PIDController(BaseController):
    def __init__(self, error='CrossTrackError'):
        super(PIDController, self).__init__(error)

    def reset_params(self):
        with self.path_lock:
            self.finish_threshold = float(rospy.get_param("/pid/finish_threshold", 0.2))
            self.exceed_threshold = float(rospy.get_param("/pid/exceed_threshold", 4.0))
            self.waypoint_lookahead = float(rospy.get_param("/pid/waypoint_lookahead", 0.6))

            self.kp = float(rospy.get_param("/pid/kp", 0.30))
            self.kd = float(rospy.get_param("/pid/kd", 0.45))
	    self.prev_error=0
	    self.prev_time = time.time()
            self.error = rospy.get_param("/pid/error", "CrossTrackError")

    def reset_state(self):
        pass

    def get_control(self, pose, index):
        # TODO 2.1: INSERT CODE HERE. Don't delete this line.
        #
        # Compute the next control using the PD control strategy.
        # Consult the spec for details on how PD control works.
	err = self.get_error(pose,index)[1]
	ref = self.get_reference_pose(index)
	ref_angle = ref[2]
	V = ref[3]
	# Analytic deriv:
	deriv = V*np.sin(pose[2]-ref_angle)
	# Finite difference deriv:
	#curr_time = time.time()
	#deriv = (err-self.prev_error)/float(curr_time-self.prev_time)
	#self.prev_error = err
	#self.prev_time = curr_time
	control = np.array([V,-1.0*(self.kp*err + self.kd*deriv)])
	return control

