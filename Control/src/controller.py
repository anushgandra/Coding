import numpy as np
import threading


class BaseController(object):
    def __init__(self, error='CrossTrackError'):
        self.path_lock = threading.RLock()
        self.path = np.array([])
        self._is_ready = False

        self.error = error

        self.reset_params()
        self.reset_state()

        self.waypoint_lookahead = 0.5 # Average distance from the current
                                      # reference pose to lookahed. You may need
                                      # this parameter in implementations later.

    def reset_params(self):
        # updating parameters from the ros parameter
        # overwrite this function in the child class
        raise NotImplementedError

    def reset_state(self):
        # resets the controller's internal state
        # overwrite this function in the child class
        raise NotImplementedError

    def get_control(self, pose, index):
        '''
        get_control - computes the control action given the current pose of
                      the car and an index into the reference trajectory
        input:
            pose - the vehicle's current pose [x, y, heading]
            index - an integer index into the reference path
        output:
            control - [target velocity, target steering angle]
        '''
        # overwrite this function in the child class
        raise NotImplementedError

    ############################################################
    # Helper Functions
    ############################################################

    def is_ready(self):
        # returns whether controller is ready to begin tracking trajectory.
        return self._is_ready

    def get_reference_pose(self, index):
        # returns the pose from the reference path at the reference index.
        with self.path_lock:
            assert len(self.path) > index
            return self.path[index]

    def set_path(self, path):
        # sets the reference trajectory, implicitly resets internal state
        with self.path_lock:
            self.path = np.array(
                [np.array([path[i].x, path[i].y, path[i].h, path[i].v])
                    for i in range(len(path))])
            self.reset_state()
            self._is_ready = True
            # average distance between path waypoints
            self.waypoint_diff = np.average(
                np.linalg.norm(np.diff(self.path[:, :2], axis=0), axis=1))

    def get_reference_index(self, pose):
        '''
        given the current pose, finds the the "reference index" i of the
            controller's path that will be used as the next control target.
        input:
            pose - current pose of the car, represented as [x, y, heading]
        output:
            i - index
        '''
        # TODO 1.1: INSERT CODE HERE. Don't modify / delete this line
        #
        # Determine a strategy for selecting a reference point
        # in the path. One option is to:
        #   STEP1. find the nearest reference point to the current_pose, p_close
        #   STEP2. chose the next point after p_close, some distance away
        #          along the path.
        #          Here, "some distance" can be defined as a parameter
        #          e.g. self.waypoint_lookahead.
        #
        # Note: this method must be computationally efficient
        # as it is running directly in the control loop.
	last_ind = np.shape(self.path)[0]
	
        diff = pose[0:-1] - self.path[:,0:2] 
	dist = np.linalg.norm(diff,axis=-1)
	minimum = np.argmin(dist)
	ahead = np.where(dist[minimum+1:]>=self.waypoint_lookahead)[0]	
	if (ahead != []):
		ahead = ahead[0]+minimum+1
	else:
		ahead = last_ind-1
	
	'''
	ahead = -1
	dist = 100000000000000000000000000.0
	minimum=-1
	for i in range(last_ind):
		temp = ((pose[0]-self.path[i,0])**2 + (pose[1]-self.path[i,1])**2)**0.5
		if (temp<dist):
			temp = dist
			minimum = i

	for i in range(minimum+1,last_ind):
		temp = ((pose[0]-self.path[i,0])**2 + (pose[1]-self.path[i,1])**2)**0.5
		if(temp>=self.waypoint_lookahead):
			ahead = i

	if (ahead==-1):
		ahead = last_ind-1
	'''
	
	return ahead
	

    def path_complete(self, pose, error):
        '''
        path_complete computes whether the vehicle has completed the path
            based on whether the reference index refers to the final point
            in the path and whether e_x is below the finish_threshold
            or e_y exceeds an 'exceed threshold'.
        input:
            pose - current pose of the vehicle [x, y, heading]
            error - error vector [e_x, e_y]
        output:
            is_path_complete - boolean stating whether the vehicle has
                reached the end of the path
        '''
        err_l2 = np.linalg.norm(error)
        return ((self.get_reference_index(pose) == (len(self.path) - 1)
            and err_l2 < self.finish_threshold)
            or (err_l2 > self.exceed_threshold))


    def get_error(self, pose, index):
        '''
        Computes the error vector for a given pose and reference index.
        input:
            pose - pose of the car [x, y, heading]
            index - integer corresponding to the reference index into the
                reference path
        output:
            e_p - error vector [e_x, e_y]
        '''
        if self.error == 'CrossTrackError':
            return self._get_cross_track_error(pose, index)
        else:
            return self._get_alternative_error(pose, index)

    def _get_cross_track_error(self, pose, index):
        # TODO 1.2: INSERT  CODE HERE. Don't modify / delete this line
        #
        # Use the method described in the handout to compute the error vector.
        # Be careful about the order in which you subtract the car's pose
        # from the reference pose.
	ref = self.get_reference_pose(index)
	diff = pose[0:2]-ref[0:2]
	theta = ref[2]
	ex = np.cos(theta)*diff[0] + np.sin(theta)*diff[1]
	ey = -np.sin(theta)*diff[0] + np.cos(theta)*diff[1]
	return [ex,ey]		
        

    def _get_alternative_error(self, pose, index):
        # TODO E.A: INSERT CODE HERE. Don't modify / delete this line
        #
        # (Extra Credit) implement an alternative error definition
        raise NotImplementedError

