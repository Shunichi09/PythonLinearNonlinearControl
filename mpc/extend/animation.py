import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import matplotlib.font_manager as fon
import sys
import math

# default setting of figures
plt.rcParams["mathtext.fontset"] = 'stix' # math fonts
plt.rcParams['xtick.direction'] = 'in' # x axis in
plt.rcParams['ytick.direction'] = 'in' # y axis in 
plt.rcParams["font.size"] = 10
plt.rcParams['axes.linewidth'] = 1.0 # axis line width
plt.rcParams['axes.grid'] = True # make grid

def coordinate_transformation_in_angle(positions, base_angle):
    '''
    Transformation the coordinate in the angle

    Parameters
    -------
    positions : numpy.ndarray
        this parameter is composed of xs, ys 
        should have (2, N) shape 
    base_angle : float [rad]
    
    Returns
    -------
    traslated_positions : numpy.ndarray
        the shape is (2, N)
    
    '''
    if positions.shape[0] != 2:
        raise ValueError('the input data should have (2, N)')

    positions = np.array(positions)
    positions = positions.reshape(2, -1)

    rot_matrix = [[np.cos(base_angle), np.sin(base_angle)],
                  [-1*np.sin(base_angle), np.cos(base_angle)]]

    rot_matrix = np.array(rot_matrix)
    
    translated_positions = np.dot(rot_matrix, positions)

    return translated_positions

def square_make_with_angles(center_x, center_y, size, angle):
    '''
    Create square matrix with angle line matrix(2D)
    
    Parameters
    -------
    center_x : float in meters
        the center x position of the square
    center_y : float in meters
        the center y position of the square
    size : float in meters
        the square's half-size
    angle : float in radians

    Returns
    -------
    square xs : numpy.ndarray
        lenght is 5 (counterclockwise from right-up)
    square ys : numpy.ndarray
        length is 5 (counterclockwise from right-up)
    angle line xs : numpy.ndarray
    angle line ys : numpy.ndarray
    '''

    # start with the up right points
    # create point in counterclockwise
    square_xys = np.array([[size, 0.5 * size], [-size, 0.5 * size], [-size, -0.5 * size], [size, -0.5 * size], [size, 0.5 * size]])
    trans_points = coordinate_transformation_in_angle(square_xys.T, -angle) # this is inverse type
    trans_points += np.array([[center_x], [center_y]])

    square_xs = trans_points[0, :]
    square_ys = trans_points[1, :]

    angle_line_xs = [center_x, center_x + math.cos(angle) * size]
    angle_line_ys = [center_y, center_y + math.sin(angle) * size]

    return square_xs, square_ys, np.array(angle_line_xs), np.array(angle_line_ys)


def circle_make_with_angles(center_x, center_y, radius, angle):
    '''
    Create circle matrix with angle line matrix
    
    Parameters
    -------
    center_x : float
        the center x position of the circle
    center_y : float
        the center y position of the circle
    radius : float
    angle : float [rad]
    
    Returns
    -------
    circle xs : numpy.ndarray
    circle ys : numpy.ndarray
    angle line xs : numpy.ndarray
    angle line ys : numpy.ndarray
    '''

    point_num = 100 # 分解能

    circle_xs = []
    circle_ys = []

    for i in range(point_num + 1):
        circle_xs.append(center_x + radius * math.cos(i*2*math.pi/point_num))
        circle_ys.append(center_y + radius * math.sin(i*2*math.pi/point_num))

    angle_line_xs = [center_x, center_x + math.cos(angle) * radius]
    angle_line_ys = [center_y, center_y + math.sin(angle) * radius]

    return np.array(circle_xs), np.array(circle_ys), np.array(angle_line_xs), np.array(angle_line_ys)


class AnimDrawer():
    """create animation of path and robot
    
    Attributes
    ------------
    cars : 
    anim_fig : figure of matplotlib
    axis : axis of matplotlib

    """
    def __init__(self, objects):
        """
        Parameters
        ------------
        objects : list of objects

        Notes
        ---------
        lead_history_states, lead_history_predict_states, traj_ref, history_traj_ref, history_angle_ref
        """
        self.lead_car_history_state = objects[0]
        self.lead_car_history_predict_state = objects[1]
        self.traj = objects[2]
        self.history_traj_ref = objects[3]
        self.history_angle_ref = objects[4]
        
        self.history_xs = [self.lead_car_history_state[:, 0]]
        self.history_ys = [self.lead_car_history_state[:, 1]]
        self.history_ths = [self.lead_car_history_state[:, 2]]

        # setting up figure
        self.anim_fig = plt.figure(dpi=150)
        self.axis = self.anim_fig.add_subplot(111)

        # imgs
        self.car_imgs = []
        self.traj_imgs = []
        self.predict_imgs = []

    def draw_anim(self, interval=50):
        """draw the animation and save

        Parameteres
        -------------
        interval : int, optional
            animation's interval time, you should link the sampling time of systems
            default is 50 [ms]
        """
        self._set_axis()
        self._set_img()

        self.skip_num = 1
        frame_num = int((len(self.history_xs[0])-1) / self.skip_num)

        animation = ani.FuncAnimation(self.anim_fig, self._update_anim, interval=interval, frames=frame_num)

        # self.axis.legend()
        print('save_animation?')
        shuold_save_animation = int(input())

        if shuold_save_animation: 
            print('animation_number?')
            num = int(input())
            animation.save('animation_{0}.mp4'.format(num), writer='ffmpeg')
            # animation.save("Sample.gif", writer = 'imagemagick') # gif保存

        plt.show()

    def _set_axis(self):
        """ initialize the animation axies
        """
        # (1) set the axis name
        self.axis.set_xlabel(r'$\it{x}$ [m]')
        self.axis.set_ylabel(r'$\it{y}$ [m]')
        self.axis.set_aspect('equal', adjustable='box')

        LOW_MARGIN = 5
        HIGH_MARGIN = 5

        self.axis.set_xlim(np.min(self.history_xs) - LOW_MARGIN, np.max(self.history_xs) + HIGH_MARGIN)
        self.axis.set_ylim(np.min(self.history_ys) - LOW_MARGIN, np.max(self.history_ys) + HIGH_MARGIN)
        
    def _set_img(self):
        """ initialize the imgs of animation
            this private function execute the make initial imgs for animation
        """
        # object imgs
        obj_color_list = ["k", "k", "m", "m"]
        obj_styles = ["solid", "solid", "solid", "solid"]

        for i in range(len(obj_color_list)):
            temp_img, = self.axis.plot([], [], color=obj_color_list[i], linestyle=obj_styles[i])
            self.car_imgs.append(temp_img)
        
        traj_color_list = ["k", "b"]

        for i in range(len(traj_color_list)):
            temp_img, = self.axis.plot([],[], color=traj_color_list[i], linestyle="dashed")
            self.traj_imgs.append(temp_img)

        temp_img, = self.axis.plot([],[], ".", color="m")
        self.traj_imgs.append(temp_img)

        # predict
        for _ in range(2 * len(self.history_angle_ref[0])):
            temp_img, = self.axis.plot([],[], color="g", linewidth=0.5) # point
            # temp_img, = self.axis.plot([],[], ".", color="g", linewidth=0.5) # point
            self.predict_imgs.append(temp_img)
    
    def _update_anim(self, i):
        """the update animation
        this function should be used in the animation functions

        Parameters
        ------------
        i : int
            time step of the animation
            the sampling time should be related to the sampling time of system

        Returns
        -----------
        object_imgs : list of img
        traj_imgs : list of img
        """
        i = int(i * self.skip_num)

        # self._draw_set_axis(i)
        self._draw_car(i)
        self._draw_traj(i)
        # self._draw_prediction(i)

        return self.car_imgs, self.traj_imgs, self.predict_imgs, 

    def _draw_set_axis(self, i):
        """
        """
        # (2) set the xlim and ylim
        LOW_MARGIN = 20
        HIGH_MARGIN = 20
        OVER_LOOK = 50
        self.axis.set_xlim(np.min(self.history_xs[0][i : i + OVER_LOOK]) - LOW_MARGIN, np.max(self.history_xs[0][i : i + OVER_LOOK]) + HIGH_MARGIN)
        self.axis.set_ylim(np.min(self.history_ys[0][i : i + OVER_LOOK]) - LOW_MARGIN, np.max(self.history_ys[0][i : i + OVER_LOOK]) + HIGH_MARGIN)         

    def _draw_car(self, i):
        """
        This private function is just divided thing of
        the _update_anim to see the code more clear

        Parameters
        ------------
        i : int
            time step of the animation
            the sampling time should be related to the sampling time of system
        """
        # cars
        object_x, object_y, angle_x, angle_y = square_make_with_angles(self.history_xs[0][i],
                                                                        self.history_ys[0][i], 
                                                                        5.0,
                                                                        self.history_ths[0][i])

        self.car_imgs[0].set_data([object_x, object_y])
        self.car_imgs[1].set_data([angle_x, angle_y])

    def _draw_traj(self, i):
        """
        This private function is just divided thing of
        the _update_anim to see the code more clear

        Parameters
        ------------
        i : int
            time step of the animation
            the sampling time should be related to the sampling time of system
        """
        # car
        self.traj_imgs[0].set_data(self.history_xs[0][:i], self.history_ys[0][:i])

        # all traj_ref
        self.traj_imgs[1].set_data(self.traj[0, :], self.traj[1, :])

        # traj_ref
        # self.traj_imgs[2].set_data(self.history_traj_ref[i][0, :], self.history_traj_ref[i][1, :])

    def _draw_prediction(self, i):
        """draw prediction

        Parameters
        ------------
        i : int
            time step of the animation
            the sampling time should be related to the sampling time of system
        """

        for j in range(0, len(self.history_angle_ref[0]), 4):
            fix_j = j * 2
            object_x, object_y, angle_x, angle_y =\
                 circle_make_with_angles(self.lead_car_history_predict_state[i][j, 0],
                 self.lead_car_history_predict_state[i][j, 1], 1.,
                 self.lead_car_history_predict_state[i][j, 2])

            self.predict_imgs[fix_j].set_data(object_x, object_y)
            self.predict_imgs[fix_j + 1].set_data(angle_x, angle_y)