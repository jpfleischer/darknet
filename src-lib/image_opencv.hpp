#pragma once

#include "darknet_internal.hpp"


/// Hide a C++ cv::Mat object as a C style @p void* pointer.
typedef void* mat_cv;

/// Hide a C++ cv::VideoCapture object as a C style @p void* pointer.
typedef void* cap_cv;

/// Hide a C++ cv::VideoWriter object as a C style @p void* pointer.
typedef void* write_cv;


/** Load the given image using OpenCV and return it as a @p "C" style @ref mat_cv @p void* pointer.  Converts the image
 * from the usual OpenCV BGR format to RGB.  Remember to free the image with @ref release_mat().
 */
mat_cv * load_image_mat_cv(const char * const filename, int flag);


/** Similar to @ref load_image_mat_cv() but is explicit about the image channels and returns a Darknet-style
 * @ref Darknet::Image.
 *
 * Channels should be @p 0 (colour), @p 1 (grayscale) or @p 3 (colour).  This uses @ref load_image_mat_cv() so the
 * channels will be converted from BGR to RGB.
 */
Darknet::Image load_image_cv(char *filename, int channels);



int get_width_mat(mat_cv *mat);
int get_height_mat(mat_cv *mat);

/// Frees the @p cv::Mat object allocated in @ref load_image_mat_cv().
void release_mat(mat_cv **mat);

/** Convert an OpenCV @p cv::Mat object to @ref Darknet::Image.  The @p cv::Mat is expected to already have been
 * converted from BGR to RGB.  And the result @ref Darknet::Image floats will be normalized between @p 0.0 and @p 1.0.
 */
Darknet::Image mat_to_image(cv::Mat mat);

/// Do not use.  @todo This needs to be removed.
Darknet::Image mat_to_image_cv(mat_cv *mat);

/** Convert the usual @ref Darknet::Image format to OpenCV @p cv::Mat.  The mat object will be in RGB format, not BGR.
 * @see @ref mat_to_image()
 */
cv::Mat image_to_mat(Darknet::Image img);

// Window
void create_window_cv(char const* window_name, int full_screen, int width, int height);
//void make_window(char *name, int w, int h, int fullscreen); -- use create_window_cv() instead
void show_image_cv(Darknet::Image p, const char *name);
//void show_image_cv_ipl(mat_cv *disp, const char *name);
void show_image_mat(mat_cv *mat_ptr, const char *name);

// Video Writer
write_cv *create_video_writer(char *out_filename, char c1, char c2, char c3, char c4, int fps, int width, int height, int is_color);
void write_frame_cv(write_cv *output_video_writer, mat_cv *mat);
void release_video_writer(write_cv **output_video_writer);


// Video Capture
cap_cv* get_capture_video_stream(const char *path);
cap_cv* get_capture_webcam(int index);
void release_capture(cap_cv* cap);

mat_cv* get_capture_frame_cv(cap_cv *cap);
int get_stream_fps_cpp_cv(cap_cv *cap);
double get_capture_property_cv(cap_cv *cap, int property_id);
double get_capture_frame_count_cv(cap_cv *cap);
int set_capture_property_cv(cap_cv *cap, int property_id, double value);
int set_capture_position_frame_cv(cap_cv *cap, int index);

// ... Video Capture
Darknet::Image get_image_from_stream_cpp(cap_cv *cap);
Darknet::Image get_image_from_stream_resize(cap_cv *cap, int w, int h, int c, mat_cv** in_img, int dont_close);
Darknet::Image get_image_from_stream_letterbox(cap_cv *cap, int w, int h, int c, mat_cv** in_img, int dont_close);
void consume_frame(cap_cv *cap);

// Image Saving
void save_cv_png(mat_cv *img, const char *name);
void save_cv_jpg(mat_cv *img, const char *name);

// Draw Detection
void draw_detections_cv_v3(mat_cv* show_img, Darknet::Detection *dets, int num, float thresh, char **names, int classes, int ext_output);

// Data augmentation
Darknet::Image image_data_augmentation(mat_cv* mat, int w, int h,
    int pleft, int ptop, int swidth, int sheight, int flip,
    float dhue, float dsat, float dexp,
    int gaussian_noise, int blur, int num_boxes, int truth_size, float *truth);

// blend two images with (alpha and beta)
void blend_images_cv(Darknet::Image new_img, float alpha, Darknet::Image old_img, float beta);

// bilateralFilter bluring
Darknet::Image blur_image(Darknet::Image src_img, int ksize);

// Show Anchors
void show_anchors(int number_of_boxes, int num_of_clusters, float *rel_width_height_array, model anchors_data, int width, int height);

void show_opencv_info();
