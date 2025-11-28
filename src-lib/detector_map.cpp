#include "darknet_internal.hpp"


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();

	/* ********************* */
	/* PROBABILITY STRUCTURE */
	/* ********************* */
	/// BoxProbability is used to match probabilities to ground truth.  This happens in the "analysis" thread.
	struct BoxProbability
	{
		Darknet::Box bb				= {0.0f, 0.0f, 0.0f, 0.0f}; ///< bounding box
		float probability			= 0.0f;						///< probability (score)
		int class_id				= -1;
		bool matched_ground_truth	= false;					///< @p TRUE if matched to some ground truth (greedy best-IoU of same class), else @p FALSE
		int unique_truth_index		= -1;						///< global index of matched ground truth, to prevent double counting
	};

	/* ******************* */
	/* WORK UNIT STRUCTURE */
	/* ******************* */
	// A single unit of work ("image") to be performed by the loading thread, prediction thread, and analysis thread.
	struct WorkUnit
	{
		// image resized to the network dimensions
		Darknet::Image img = {0};

		// results of calling Darknet's predict()
		Darknet::Detection * predictions = nullptr;
		int number_of_predictions = 0;

		// results read from .txt annotation file
		box_label * ground_truth_labels = nullptr;
		int number_of_ground_truth_labels = 0;
	};

	/** Information which needs to be shared between threads.  Group it together in a structure and only share 1 struct
	 * instead of many individual fields passed around between threads.
	 */
	struct SharedInfo
	{
		Darknet::Network net;

		/// A copy of the last YOLO layer in the network.
		Darknet::Layer output_layer;

		float iou_threshold			= 0.5f;		///< user-selected IoU threshold; e.g., see command-line parm "-iou_thresh 0.5"
		float thresh_calc_avg_iou	= 0.25f; 	///< @todo what is this?  e.g., see command-line parm "-thresh 0.25"
		float detection_threshold	= 0.005f;	///< detection threshold
		float nms					= 0.45f;
		float avg_iou				= 0.0f;
		int tp_for_thresh			= 0;		///< diagnostic TP at thresh_calc_avg_iou (across all classes)
		int fp_for_thresh			= 0;		///< diagnostic FP at thresh_calc_avg_iou (across all classes)
		int unique_truth_count		= 0;		///< @todo what is this?

		std::vector<float> avg_iou_per_class;
		std::vector<int> tp_for_thresh_per_class;
		std::vector<int> fp_for_thresh_per_class;

		/** All of the predictions across the entire dataset.  Obviously, this can grow to be quite big.  But this happens in
		 * the analysis thread, which is not the bottleneck, so we can ignore trying to apply performance optimizations.
		 */
		std::vector<BoxProbability> box_probabilities;

		/** When the loading threads are faster than the prediction thread, we need to insert some pauses in between image
		 * loading.  These pauses will increase or decrease in length to accomodate for faster computers, faster drives,
		 * image sizes, resizing needs, etc.  Range will be from 5 to 150 milliseconds.
		 * @see @ref max_sleep_time_in_milliseconds
		 */
		std::atomic<int> length_of_loading_pause_in_milliseconds = 5;

		/** When the prediction thread is faster than the loading thread, we need to decrease the pause length for loading
		 * and increase the pause length between predictions.  Range will be from 5 to 150 milliseconds.
		 * @see @ref max_sleep_time_in_milliseconds
		 */
		std::atomic<int> length_of_prediction_pause_in_milliseconds = 5;

		/// The maximum amount of time a thread will sleep when it is starved or needs to pause.
		const int max_sleep_time_in_milliseconds = 150;

		/// The total number of classes in this neural network.
		size_t number_of_classes = 0;

		/** This value is important since the validation images are split into multiple input queues.  This value is the
		 * @em total of all input images, which the threads use to determine when they've finished processing all images.
		 */
		size_t total_number_of_validation_images = 0;

		/// Note the maximum work queue size, where all loading threads fill up a single work queue.  @see @ref max_work_queue_size
		const size_t number_of_loading_threads_to_start = std::clamp(std::thread::hardware_concurrency() / 3, 1U, 3U);

		/** This is the total number of images loaded into the work queue waiting to be processed.  The larger the number,
		 * the more RAM is needed (mostly to store the images).  @see @ref number_of_loading_threads_to_start
		 */
		const size_t max_work_queue_size = number_of_loading_threads_to_start * 50;

		/** This is in a vector because we often want to start multiple loading threads, and each thread requires a unique
		 * set of validation image filenames.  @see @ref number_of_loading_threads_to_start
		 */
		std::vector<Darknet::SStr> validation_image_filenames;

		/** Remember exactly how many ground truths we have for each class.  The key is the class ID, the value is the
		 * counter for that specific class.
		 */
		std::map<int, size_t> ground_truth_counts;

		/// Similar to @ref ground_truth_counts but for predictions.
		std::map<int, size_t> prediction_counts;

		/// Work is placed here once it has *finished* being loaded and prior to predictions.  @see @ref work_ready_for_predictions_mutex
		std::map<std::string, WorkUnit> work_ready_for_predictions;

		/// Work is placed here once it has *finished* predictions and is ready for calculations.  @see @ref work_ready_for_calculations_mutex
		std::map<std::string, WorkUnit> work_ready_for_calculations;

		/// Must lock prior to modifying the STL container @ref work_ready_for_predictions.
		std::mutex work_ready_for_predictions_mutex;

		/// Must lock prior to modifying the STL container @ref work_ready_for_calculations.
		std::mutex work_ready_for_calculations_mutex;

		/// The total number of images which have been loaded from disk.
		std::atomic<size_t> count_load_performed	= 0;

		/// The number of work units that remain on the internal work queue waiting for Darknet predict().
		std::atomic<size_t> count_predict_internal	= 0;

		/// The total number of images which have been processed by Darknet predict().
		std::atomic<size_t> count_predict_performed	= 0;

		/// The number of work units that remain on the internal work queue waiting for analysis.
		std::atomic<size_t> count_analyze_internal	= 0;

		/// The total number of images which have completed the analysis.
		std::atomic<size_t> count_analyze_performed	= 0;

		/// The number of times image loading had to be paused because prediction queue had reached the maximum work units.  @see @ref max_work_queue_size
		std::atomic<size_t> count_loading_paused	= 0;

		/// The number of times predictions had to be paused because no images had been loaded and put in the work queue.
		std::atomic<size_t> count_predict_starved	= 0;

		/// The number of times analysis had to be paused because predictions hadn't yet processed images.
		std::atomic<size_t> count_analyze_starved	= 0;
	};


	/* ************************ */
	/* LOAD LOAD LOAD LOAD LOAD */
	/* ************************ */

	/** Load images.  Try and do the least amount of work possible here since the loading threads are important to
	 * keeping the prediction thread fed.  @note This is called on a secondary thread!
	 */
	void detector_map_loading_thread(const size_t loading_thread_id, SharedInfo & shared_info)
	{
		/* THIS IS CALLED ON A SECONDARY THREAD!
		 *
		 * Multiple instances of this may exist at the same time, since we typically have 2 or more loading threads.
		 *
		 * [[[*** BEWARE! ***]]]  There may be multiple copies of the loading thread running.  Only values in WorkUnit may be
		 * modified.  Values in other places, like SharedInfo, can only be modified with mutex protection or with std::atomic.
		 */

		try
		{
			cfg_and_state.set_thread_name("detector map image loading thread #" + std::to_string(loading_thread_id));

			const int w = shared_info.net.w;
			const int h = shared_info.net.h;
			const int c = shared_info.net.c;

			for (auto iter = shared_info.validation_image_filenames[loading_thread_id].begin(); iter != shared_info.validation_image_filenames[loading_thread_id].end() and cfg_and_state.must_immediately_exit == false; iter ++)
			{
				if (shared_info.work_ready_for_predictions.size() >= shared_info.max_work_queue_size)
				{
					// increase the length of time we pause since images are loading faster than we can process them
					shared_info.length_of_loading_pause_in_milliseconds = std::clamp(shared_info.length_of_loading_pause_in_milliseconds * 2, 5, shared_info.max_sleep_time_in_milliseconds);
					// decrease the length of time the prediction thread sleeps
					shared_info.length_of_prediction_pause_in_milliseconds = std::clamp(shared_info.length_of_prediction_pause_in_milliseconds / 2, 5, shared_info.max_sleep_time_in_milliseconds);

					if (cfg_and_state.is_trace)
					{
						*cfg_and_state.output << "=> pause image loading for " << shared_info.length_of_loading_pause_in_milliseconds << " milliseconds since map already contains " << shared_info.work_ready_for_predictions.size() << " items" << std::endl;
					}
					while (shared_info.work_ready_for_predictions.size() >= shared_info.max_work_queue_size and cfg_and_state.must_immediately_exit == false)
					{
						shared_info.count_loading_paused ++;
						std::this_thread::sleep_for(std::chrono::milliseconds(shared_info.length_of_loading_pause_in_milliseconds));
					}
				}

				shared_info.count_load_performed ++;
				std::filesystem::path fn = *iter;
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "-> " << shared_info.count_load_performed << ": loading " << fn.string() << std::endl;
				}

				// load the image and resize it to match the network dimensions
				WorkUnit work;
				work.img = Darknet::load_image(iter->c_str(), w, h, c);

				std::lock_guard lock(shared_info.work_ready_for_predictions_mutex);
				shared_info.work_ready_for_predictions[*iter] = work;
			}

			cfg_and_state.del_thread_name();
		}
		catch (const std::exception & e)
		{
			darknet_fatal_error(DARKNET_LOC, "exception caught while loading images for map: %s", e.what());
		}

		return;
	}


	/* *************************************** */
	/* PREDICT PREDICT PREDICT PREDICT PREDICT */
	/* *************************************** */

	/// Get Darknet predictions for each image.  @note This is called on a secondary thread!
	void detector_map_prediction_thread(SharedInfo & shared_info)
	{
		// THIS IS CALLED ON A SECONDARY THREAD!

		try
		{
			cfg_and_state.set_thread_name("map prediction thread");

			std::map<std::string, WorkUnit> work_to_do;
			while (shared_info.count_predict_performed < shared_info.total_number_of_validation_images and cfg_and_state.must_immediately_exit == false)
			{
				if (work_to_do.empty())
				{
					if (shared_info.work_ready_for_predictions.empty())
					{
						// loading is too slow, try and make it more responsive
						shared_info.length_of_loading_pause_in_milliseconds = std::clamp(shared_info.length_of_loading_pause_in_milliseconds / 2, 5, shared_info.max_sleep_time_in_milliseconds);
						// at the same time, increase the length of prediction pauses
						shared_info.length_of_prediction_pause_in_milliseconds = std::clamp(shared_info.length_of_prediction_pause_in_milliseconds * 2, 5, shared_info.max_sleep_time_in_milliseconds);

						if (cfg_and_state.is_trace)
						{
							*cfg_and_state.output << "=> prediction thread is starved, pausing for " << shared_info.length_of_prediction_pause_in_milliseconds << " milliseconds" << std::endl;
						}

						shared_info.count_predict_starved ++;
						std::this_thread::sleep_for(std::chrono::milliseconds(shared_info.length_of_prediction_pause_in_milliseconds));

						continue;
					}

					if (cfg_and_state.is_trace)
					{
						*cfg_and_state.output << "=> swapping in " << shared_info.work_ready_for_predictions.size() << " new work units for predictions thread" << std::endl;
					}

					std::lock_guard lock(shared_info.work_ready_for_predictions_mutex);
					shared_info.work_ready_for_predictions.swap(work_to_do);
					shared_info.count_predict_internal = work_to_do.size();
				}

				for (auto & [fn, work] : work_to_do)
				{
					if (cfg_and_state.must_immediately_exit)
					{
						break;
					}

					shared_info.count_predict_performed ++;
					if (cfg_and_state.is_trace)
					{
						*cfg_and_state.output << "-> " << shared_info.count_predict_performed << ": predicting with " << fn << std::endl;
					}

					network_predict(shared_info.net, work.img.data);
					const int image_width	= work.img.w;
					const int image_height	= work.img.h;
					Darknet::free_image(work.img);

					const float hierarchy_threshold = 0.5f;
					if (shared_info.net.letter_box == LETTERBOX_DATA) /// @todo we should eventually get rid of letterbox
					{
						work.predictions = get_network_boxes(&shared_info.net, image_width, image_height, shared_info.detection_threshold, hierarchy_threshold, 0, 1, &work.number_of_predictions, shared_info.net.letter_box);
					}
					else
					{
						work.predictions = get_network_boxes(&shared_info.net, 1, 1, shared_info.detection_threshold, hierarchy_threshold, 0, 0, &work.number_of_predictions, shared_info.net.letter_box);
					}

					if (shared_info.nms)
					{
						if (shared_info.output_layer.nms_kind == DEFAULT_NMS)
						{
							do_nms_sort(work.predictions, work.number_of_predictions, shared_info.output_layer.classes, shared_info.nms);
						}
						else
						{
							diounms_sort(work.predictions, work.number_of_predictions, shared_info.output_layer.classes, shared_info.nms, shared_info.output_layer.nms_kind, shared_info.output_layer.beta_nms);
						}
					}

					std::lock_guard lock(shared_info.work_ready_for_calculations_mutex);
					shared_info.work_ready_for_calculations[fn] = work;
					shared_info.count_predict_internal --;
				}
				work_to_do.clear();
				shared_info.count_predict_internal = 0;
			}

			cfg_and_state.del_thread_name();
		}
		catch (const std::exception & e)
		{
			darknet_fatal_error(DARKNET_LOC, "exception caught while obtaining predictions for map: %s", e.what());
		}

		return;
	}


	/* ******************************************** */
	/* ANALYSIS ANALYSIS ANALYSIS ANALYSIS ANALYSIS */
	/* ******************************************** */

	/// Run mAP calculations for each image.  Ground truth annotations are also loaded here to reduce the tasks done by the image loading thread.  @note This is called on a secondary thread!
	void detector_map_calculations_thread(SharedInfo & shared_info)
	{
		// THIS IS CALLED ON A SECONDARY THREAD!

		try
		{
			cfg_and_state.set_thread_name("map calculations thread");

			std::map<std::string, WorkUnit> work_to_do;
			while (shared_info.count_analyze_performed < shared_info.total_number_of_validation_images and cfg_and_state.must_immediately_exit == false)
			{
				if (work_to_do.empty())
				{
					if (shared_info.work_ready_for_calculations.empty())
					{
						shared_info.count_analyze_starved ++;

						const int time_to_pause_in_milliseconds = std::max(shared_info.length_of_loading_pause_in_milliseconds, shared_info.length_of_prediction_pause_in_milliseconds);
						if (cfg_and_state.is_trace)
						{
							*cfg_and_state.output << "=> calculation thread is starved, pausing for " << time_to_pause_in_milliseconds << " milliseconds" << std::endl;
						}
						std::this_thread::sleep_for(std::chrono::milliseconds(time_to_pause_in_milliseconds));
						continue;
					}

					if (cfg_and_state.is_trace)
					{
						*cfg_and_state.output << "=> swapping in " << shared_info.work_ready_for_calculations.size() << " new work units for calculations thread" << std::endl;
					}
					std::lock_guard lock(shared_info.work_ready_for_calculations_mutex);
					shared_info.work_ready_for_calculations.swap(work_to_do);
					shared_info.count_analyze_internal = work_to_do.size();
				}

				for (auto & [fn, work] : work_to_do)
				{
					if (cfg_and_state.must_immediately_exit)
					{
						break;
					}

					shared_info.count_analyze_performed ++;
					if (cfg_and_state.is_trace)
					{
						*cfg_and_state.output << "-> " << shared_info.count_analyze_performed << ": performing calculations with " << fn << std::endl;
					}

					// load the ground truth annotations for this image
					const auto ground_truth_fn = std::filesystem::path(fn).replace_extension(".txt");
					work.ground_truth_labels = read_boxes(ground_truth_fn.string().c_str(), &work.number_of_ground_truth_labels);
					if (cfg_and_state.is_trace)
					{
						*cfg_and_state.output << "-> " << shared_info.count_load_performed << ": loading " << ground_truth_fn.string() << " (" << work.number_of_ground_truth_labels << " ground truth labels)" << std::endl;
					}
					for (int j = 0; j < work.number_of_ground_truth_labels and cfg_and_state.must_immediately_exit == false; ++j)
					{
						const auto & ground_truth = work.ground_truth_labels[j];
						if (ground_truth.id < 0 or ground_truth.id >= shared_info.number_of_classes)
						{
							darknet_fatal_error(DARKNET_LOC, "invalid ground truth: class id #%d at line #%d in %s", ground_truth.id, j+1, ground_truth_fn.string().c_str());
						}

						// get an accurate count of ground truths for every class in the neural network
						shared_info.ground_truth_counts[ground_truth.id] ++;
					}

					const size_t checkpoint_box_probabilities = shared_info.box_probabilities.size();

					// go through all the predictions in this image, and try to match each one to a ground truth
					for (size_t idx = 0; idx < work.number_of_predictions and cfg_and_state.must_immediately_exit == false; idx ++)
					{
						const auto & prediction = work.predictions[idx];

						for (int class_id = 0; class_id < shared_info.number_of_classes and cfg_and_state.must_immediately_exit == false; class_id ++)
						{
							const float probability = prediction.prob[class_id];
							if (probability <= 0.0f)
							{
								// this prediction does not include this class
								continue;
							}

							shared_info.prediction_counts[class_id] ++;

							BoxProbability bp;
							bp.bb			= prediction.bbox;
							bp.probability	= probability;
							bp.class_id		= class_id;
							shared_info.box_probabilities.push_back(bp);

							auto & box_probability = *shared_info.box_probabilities.rbegin();

							// see which ground truth best matches this prediction

							int truth_index = -1;
							float best_iou = 0.0f;
							for (int j = 0; j < work.number_of_ground_truth_labels and cfg_and_state.must_immediately_exit == false; j++)
							{
								const auto & ground_truth = work.ground_truth_labels[j];

								if (ground_truth.id != bp.class_id)
								{
									continue;
								}

								// the classes match, so now figure out the IoU between the prediction and the ground truth

								const Darknet::Box box = {ground_truth.x, ground_truth.y, ground_truth.w, ground_truth.h};
								const float current_iou = box_iou(prediction.bbox, box);
								if (current_iou > shared_info.iou_threshold and current_iou > best_iou)
								{
									best_iou = current_iou;
									truth_index = shared_info.unique_truth_count + j;
								}
							}

							// remember the best IoU
							if (truth_index > -1)
							{
								box_probability.matched_ground_truth = true;
								box_probability.unique_truth_index = truth_index;
							}

							// calc avg IoU, true-positives, false-positives for required Threshold
							if (probability > shared_info.thresh_calc_avg_iou)
							{
								bool found = false;

								/* Mirror the old logic: only compare against detections from this image, i.e., those added since
								 * checkpoint_box_probabilities.
								 * shared_info.box_probabilities.size() is the *current* global count.  We stop at size() - 1 to avoid
								 * comparing the detection to itself.
								 */
								const size_t current_count = shared_info.box_probabilities.size();

								if (current_count > checkpoint_box_probabilities)
								{
									for (size_t z = checkpoint_box_probabilities; z < current_count - 1 and cfg_and_state.must_immediately_exit == false; ++z)
									{
										if (shared_info.box_probabilities[z].unique_truth_index == truth_index)
										{
											found = true;
											break;
										}
									}
								}

								if (truth_index > -1 and found == false)
								{
									shared_info.avg_iou += best_iou;
									shared_info.tp_for_thresh ++;
									shared_info.avg_iou_per_class[class_id] += best_iou;
									shared_info.tp_for_thresh_per_class[class_id]++;
								}
								else
								{
									shared_info.fp_for_thresh ++;
									shared_info.fp_for_thresh_per_class[class_id] ++;
								}
							}
						}
					}

					shared_info.unique_truth_count += work.number_of_ground_truth_labels;
					free(work.ground_truth_labels);
					work.ground_truth_labels = nullptr;
					free_detections(work.predictions, work.number_of_predictions);
					work.predictions = nullptr;

					shared_info.count_analyze_internal --;
				}

				work_to_do.clear();
				shared_info.count_analyze_internal = 0;
			}

			cfg_and_state.del_thread_name();
		}
		catch (const std::exception & e)
		{
			darknet_fatal_error(DARKNET_LOC, "exception caught while calculating results for map: %s", e.what());
		}

		return;
	}
}


/* ******************** */
/* DETECTOR MAP COMMAND */
/* ******************** */

float validate_detector_map(const char * datacfg, const char * cfgfile, const char * weightfile, float thresh_calc_avg_iou, const float iou_thresh, const int map_points, int letter_box, Darknet::Network * existing_net)
{
	/* This function is called in 2 situations:
	 *
	 *		1) During training every once in a while to calculate mAP%
	 *
	 *		2) Manually from the CLI when running a command such as the following:
	 *
	 *				darknet detector map LegoGears.cfg LegoGears_best.weights LegoGears.data
	 *
	 * This re-write of validate_detector_map() was introduced in v5.1.  The previous function was deleted.
	 */

	TAT(TATPARMS);

	*cfg_and_state.output << "Calculating mAP (mean average precision) with threshold " << thresh_calc_avg_iou << " and IoU threshold " << iou_thresh << "." << std::endl;

	SharedInfo shared_info;

	// load the network, or re-use the network already loaded
	list * options = read_data_cfg(datacfg);
	std::string validation_filename = option_find_str(options, "valid", nullptr);
	if (existing_net) // if we're being called in the middle of training a network
	{
		const char * train_images = option_find_str(options, "train", nullptr);
		validation_filename = option_find_str(options, "valid", train_images);
		shared_info.net = *existing_net;
		free_network_recurrent_state(*existing_net);
	}
	else
	{
		shared_info.net = parse_network_cfg_custom(cfgfile, 1, 1); // set batch=1
		if (weightfile)
		{
			load_weights(&shared_info.net, weightfile);
		}
		fuse_conv_batchnorm(shared_info.net);
		calculate_binary_weights(&shared_info.net);
		Darknet::load_names(&shared_info.net, option_find_str(options, "names", "unknown.names"));
	}
	free_list_contents_kvp(options);
	free_list(options);

	// split the validation images into multiple sets, where each one will be given to a different thread to load from disk
	shared_info.validation_image_filenames.resize(shared_info.number_of_loading_threads_to_start);
	if (std::filesystem::exists(validation_filename))
	{
		std::ifstream ifs(validation_filename);
		std::string line;
		while (std::getline(ifs, line) and cfg_and_state.must_immediately_exit == false)
		{
			std::filesystem::path path = line;
			if (std::filesystem::exists(path) == false)
			{
				darknet_fatal_error(DARKNET_LOC, "%s line #%ul: validation image filename is invalid: \"%s\"", validation_filename.c_str(), shared_info.total_number_of_validation_images + 1, path.string().c_str());
			}
			path.replace_extension(".txt");
			if (std::filesystem::exists(path) == false)
			{
				darknet_fatal_error(DARKNET_LOC, "%s line #%ul: validation image does not have a corresponding .txt annotation file: \"%s\"", validation_filename.c_str(), shared_info.total_number_of_validation_images + 1, path.string().c_str());
			}

			const size_t idx = (shared_info.total_number_of_validation_images % shared_info.number_of_loading_threads_to_start);
			shared_info.validation_image_filenames[idx].insert(line);
			shared_info.total_number_of_validation_images ++;
		}
	}

	if (shared_info.total_number_of_validation_images == 0)
	{
		darknet_fatal_error(DARKNET_LOC, "no validation images available (verify %s)", validation_filename);
	}

	const size_t actual_batch_size = shared_info.net.batch * shared_info.net.subdivisions;
	if (shared_info.total_number_of_validation_images < actual_batch_size)
	{
		Darknet::display_warning_msg("Warning: there seems to be very few validation images (num=" + std::to_string(shared_info.total_number_of_validation_images) + ", batch=" + std::to_string(actual_batch_size) + ")\n");
	}

	shared_info.output_layer = shared_info.net.layers[shared_info.net.n - 1];
	for (int k = 0; k < shared_info.net.n and cfg_and_state.must_immediately_exit == false; ++k)
	{
		Darknet::Layer & lk = shared_info.net.layers[k];
		if (lk.type == Darknet::ELayerType::YOLO			or
			lk.type == Darknet::ELayerType::GAUSSIAN_YOLO	or
			lk.type == Darknet::ELayerType::REGION			)
		{
			shared_info.output_layer = lk;
			*cfg_and_state.output << "-> detection layer #" << k << " is type " << static_cast<int>(lk.type) << " (" << Darknet::to_string(lk.type) << ")" << std::endl;
		}
	}

	shared_info.nms					= 0.45f;
	shared_info.iou_threshold		= iou_thresh;
	shared_info.detection_threshold	= 0.005f;
	shared_info.thresh_calc_avg_iou	= thresh_calc_avg_iou;
	shared_info.number_of_classes	= shared_info.output_layer.classes;

	shared_info.avg_iou_per_class		.resize(shared_info.number_of_classes, 0.0f);
	shared_info.tp_for_thresh_per_class	.resize(shared_info.number_of_classes, 0);
	shared_info.fp_for_thresh_per_class	.resize(shared_info.number_of_classes, 0);
	for (int class_idx = 0; class_idx < shared_info.number_of_classes and cfg_and_state.must_immediately_exit == false; class_idx ++)
	{
		shared_info.ground_truth_counts		[class_idx] = 0;
		shared_info.prediction_counts		[class_idx] = 0;
	}

	*cfg_and_state.output
		<< "-> " << shared_info.total_number_of_validation_images << " validation images"
		<< " for " << shared_info.number_of_classes << " class" << (shared_info.number_of_classes == 1 ? "" : "es") << std::endl
		<< "-> " << shared_info.number_of_loading_threads_to_start << " loading thread" << (shared_info.number_of_loading_threads_to_start == 1 ? "" : "s")
		<< " with a total work queue size of " << shared_info.max_work_queue_size << " images" << std::endl;

	const auto timestamp_start = std::chrono::high_resolution_clock::now();

	/* ************************************************************************* */
	/* START THE THREADS THAT LOAD, PREDICT, AND RUNS THE NECESSARY CALCULATIONS */
	/* ************************************************************************* */

	Darknet::VThreads all_threads;
	for (size_t idx = 0; idx < shared_info.number_of_loading_threads_to_start and cfg_and_state.must_immediately_exit == false; idx ++)
	{
		*cfg_and_state.output << "-> starting loading thread #" << idx << " with " << shared_info.validation_image_filenames[idx].size() << " images" << std::endl;
		all_threads.emplace_back(detector_map_loading_thread, idx, std::ref(shared_info));
	}

	// give the loading threads time to load some images before we start the other threads
	std::this_thread::sleep_for(std::chrono::milliseconds(50));
	all_threads.emplace_back(detector_map_prediction_thread,	std::ref(shared_info));
	all_threads.emplace_back(detector_map_calculations_thread,	std::ref(shared_info));

	/* ******************************* */
	/* WAIT UNTIL ALL THREADS ARE DONE */
	/* ******************************* */

	size_t previous_count_analyze	= 0;
	auto previous_timestamp			= timestamp_start;

	bool done = false;
	while (cfg_and_state.must_immediately_exit == false and not done)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(750));

		if (shared_info.count_load_performed	>= shared_info.total_number_of_validation_images and
			shared_info.count_predict_performed	>= shared_info.total_number_of_validation_images and
			shared_info.count_analyze_performed	>= shared_info.total_number_of_validation_images)
		{
			done = true;

			// show the "full" stats since this will be the last time through the loop
			previous_count_analyze	= 0;
			previous_timestamp		= timestamp_start;
		}

		const auto now					= std::chrono::high_resolution_clock::now();
		const float nanoseconds			= std::chrono::duration_cast<std::chrono::nanoseconds>(now - previous_timestamp).count();
		const size_t images_per_second	= std::round((shared_info.count_analyze_performed	- previous_count_analyze)	/ nanoseconds * 1000000000.0f);
		previous_count_analyze			= shared_info.count_analyze_performed;
		previous_timestamp				= now;
		const int loading_percentage	= std::round(100.0f * shared_info.count_load_performed		/ shared_info.total_number_of_validation_images);
		const int predicting_percentage	= std::round(100.0f * shared_info.count_predict_performed	/ shared_info.total_number_of_validation_images);
		const int analyzing_percentage	= std::round(100.0f * shared_info.count_analyze_performed	/ shared_info.total_number_of_validation_images);
		const bool show_details			= (done or cfg_and_state.is_verbose or shared_info.count_predict_starved > 10);

		std::stringstream ss;
		ss	<< "\r"
			<< "-> " << images_per_second << " images/sec: "
			<< "loading #" << Darknet::in_colour(Darknet::EColour::kBrightWhite, int(shared_info.count_load_performed))
			<< " (" << Darknet::format_percentage(loading_percentage);

		if (show_details)
		{
			ss	<< ", pause="	<< shared_info.count_loading_paused
				<< ", time="	<< shared_info.length_of_loading_pause_in_milliseconds << " ms";
		}

		ss	<< "), predicting #" << Darknet::in_colour(Darknet::EColour::kBrightWhite, int(shared_info.count_predict_performed))
			<< " (" << Darknet::format_percentage(predicting_percentage);

		if (show_details)
		{
			ss	<< ", work="	<< shared_info.work_ready_for_predictions.size() << "+" << shared_info.count_predict_internal
				<< ", starve="	<< shared_info.count_predict_starved
				<< ", time="	<< shared_info.length_of_prediction_pause_in_milliseconds << " ms";
		}

		ss	<< "), analyzing #" << Darknet::in_colour(Darknet::EColour::kBrightWhite, int(shared_info.count_analyze_performed))
			<< " (" << Darknet::format_percentage(analyzing_percentage);

		if (show_details)
		{
			ss	<< ", work="	<< shared_info.work_ready_for_calculations.size() << "+" << shared_info.count_analyze_internal
				<< ", starve="	<< shared_info.count_analyze_starved;
		}
		ss	<< ")  "; // intentional trailing whitespace in case some fields (like "work") shrink
		if (cfg_and_state.is_verbose)
		{
			ss << std::endl;
		}

		*cfg_and_state.output << ss.str() << std::flush;
	}
	*cfg_and_state.output << std::endl;

	for (auto & t : all_threads)
	{
		t.join();
	}

	/* *********************************** */
	/* THREADS ARE DONE, PRINT THE RESULTS */
	/* *********************************** */

	if ((shared_info.tp_for_thresh + shared_info.fp_for_thresh) > 0)
	{
		shared_info.avg_iou /= (shared_info.tp_for_thresh + shared_info.fp_for_thresh);
	}

	for (int class_id = 0; class_id < shared_info.number_of_classes and cfg_and_state.must_immediately_exit == false; class_id ++)
	{
		const int denom = shared_info.tp_for_thresh_per_class[class_id] + shared_info.fp_for_thresh_per_class[class_id];
		if (denom > 0)
		{
			shared_info.avg_iou_per_class[class_id] /= denom;
		}
	}

	// Sort the array from high probability to low probability.
	//
	// With a test of 7125 entries in the array:
	//
	// - qsort() with function took:	576286 nanoseconds
	// - std::sort() with lambda took:	414231 nanoseconds
	//
	std::sort(/** @todo try this again in 2026? std::execution::par_unseq,*/ shared_info.box_probabilities.begin(), shared_info.box_probabilities.end(),
			[](const BoxProbability & lhs, const BoxProbability & rhs)
			{
				return lhs.probability > rhs.probability;
			});

	*cfg_and_state.output
		<< "detections_count=" << shared_info.box_probabilities.size()
		<< ", unique_truth_count=" << shared_info.unique_truth_count
		<< std::endl;

	double mean_average_precision = 0.0;

	// ---- Per-class AP + reporting (no TN/accuracy/specificity) ----
	for (int class_idx = 0;
		 class_idx < shared_info.number_of_classes && cfg_and_state.must_immediately_exit == false;
		 ++class_idx)
	{
		const int gt_i = static_cast<int>(shared_info.ground_truth_counts[class_idx]);

		// One truth_flags array per class, indexed by global unique_truth_index.
		// Only indices referenced by this class's detections will be touched.
		std::vector<int> truth_flags(shared_info.unique_truth_count, 0);

		int tp = 0;
		int fp = 0;

		std::vector<double> prec_hist;
		std::vector<double> rec_hist;
		prec_hist.reserve(shared_info.prediction_counts[class_idx]);
		rec_hist.reserve(shared_info.prediction_counts[class_idx]);

		// Walk detections in descending probability, but only those of this class.
		for (size_t rank = 0;
			 rank < shared_info.box_probabilities.size() && cfg_and_state.must_immediately_exit == false;
			 ++rank)
		{
			const BoxProbability &d = shared_info.box_probabilities[rank];

			if (d.class_id != class_idx)
			{
				continue;
			}

			if (d.matched_ground_truth &&
				d.unique_truth_index >= 0 &&
				d.unique_truth_index < static_cast<int>(shared_info.unique_truth_count))
			{
				if (truth_flags[d.unique_truth_index] == 0)
				{
					truth_flags[d.unique_truth_index] = 1;
					++tp; // true positive
				}
				else
				{
					++fp; // duplicate hit on same GT
				}
			}
			else
			{
				++fp; // false positive
			}

			const int det_denom = tp + fp;
			const double precision =
				(det_denom > 0) ? static_cast<double>(tp) / static_cast<double>(det_denom) : 0.0;
			const double recall =
				(gt_i > 0) ? static_cast<double>(tp) / static_cast<double>(gt_i) : 0.0;

			prec_hist.push_back(precision);
			rec_hist.push_back(recall);
		}

		// Optional diagnostic check like the old "tp+fp vs detections" print
		if (shared_info.prediction_counts[class_idx] != static_cast<size_t>(tp + fp))
		{
			*cfg_and_state.output
				<< "class_id=" << class_idx
				<< ", detections=" << shared_info.prediction_counts[class_idx]
				<< ", tp+fp=" << (tp + fp)
				<< ", tp=" << tp
				<< ", fp=" << fp
				<< std::endl;
		}

		// --- Compute AP for this class from (rec_hist, prec_hist) ---
		double avg_precision = 0.0;

		if (!prec_hist.empty())
		{
			if (map_points == 0)
			{
				// VOC2010 / AUC of precision envelope
				double last_recall = rec_hist.back();
				double last_precision = prec_hist.back();

				for (int k = static_cast<int>(prec_hist.size()) - 2;
					 k >= 0 && cfg_and_state.must_immediately_exit == false;
					 --k)
				{
					const double delta_recall = last_recall - rec_hist[k];
					last_recall = rec_hist[k];

					if (prec_hist[k] > last_precision)
					{
						last_precision = prec_hist[k];
					}

					avg_precision += delta_recall * last_precision;
				}

				// Remaining area down to recall=0
				const double delta_recall = last_recall - 0.0;
				avg_precision += delta_recall * last_precision;
			}
			else
			{
				// Sampled AP (VOC2007 11-pt, or COCO-style 101-pt)
				if (map_points < 2)
				{
					darknet_fatal_error(DARKNET_LOC, "map_points must be >= 2 (e.g., 11 or 101).");
				}

				for (int point = 0;
					 point < map_points && cfg_and_state.must_immediately_exit == false;
					 ++point)
				{
					const double cur_recall =
						(map_points == 1)
							? 0.0
							: (static_cast<double>(point) / static_cast<double>(map_points - 1));

					double cur_precision = 0.0;

					for (size_t k = 0;
						 k < prec_hist.size() && cfg_and_state.must_immediately_exit == false;
						 ++k)
					{
						if (rec_hist[k] >= cur_recall && prec_hist[k] > cur_precision)
						{
							cur_precision = prec_hist[k];
						}
					}

					avg_precision += cur_precision;
				}

				avg_precision /= map_points;
			}
		}

		// ---- Per-class stats at the SAME conf_thresh as global F1 (unchanged) ----
		const int tp_diag = shared_info.tp_for_thresh_per_class[class_idx];
		const int fp_diag = shared_info.fp_for_thresh_per_class[class_idx];
		const int fn_diag = std::max(0, gt_i - tp_diag);

		const float diag_avg_iou_at_thresh =
			(shared_info.tp_for_thresh_per_class[class_idx] +
			 shared_info.fp_for_thresh_per_class[class_idx]) > 0
				? shared_info.avg_iou_per_class[class_idx]
				: 0.0f;

		float precision_f1 = 0.0f;
		float recall_f1 = 0.0f;
		float f1 = 0.0f;

		if (tp_diag + fp_diag > 0)
		{
			precision_f1 =
				static_cast<float>(tp_diag) / static_cast<float>(tp_diag + fp_diag);
		}
		if (gt_i > 0)
		{
			recall_f1 = static_cast<float>(tp_diag) / static_cast<float>(gt_i);
		}
		if (precision_f1 + recall_f1 > 0.0f)
		{
			f1 = 2.0f * precision_f1 * recall_f1 / (precision_f1 + recall_f1);
		}

		*cfg_and_state.output
			<< Darknet::format_map_ap_row_values(
				   class_idx,										// class_id
				   shared_info.net.details->class_names[class_idx], // name
				   static_cast<float>(avg_precision),				// AP (threshold-free)
				   tp_diag,											// TP @ conf_thresh
				   0,												// TN (not shown)
				   fp_diag,											// FP @ conf_thresh
				   fn_diag,											// FN @ conf_thresh
				   gt_i,											// GT
				   f1,												// F1 @ conf_thresh
				   diag_avg_iou_at_thresh)							// diag IoU @ conf_thresh
			<< std::endl;

		// send the result of this class to the C++ side of things so we can include it the right chart
		Darknet::update_f1_in_new_charts(class_idx, f1);
		Darknet::update_accuracy_in_new_charts(class_idx, (float)avg_precision);

		mean_average_precision += avg_precision;
	}

	// Diagnostic summary (guard divisions)
	float cur_precision	= 0.0f;
	float cur_recall	= 0.0f;
	float f1_score		= 0.0f;

	const int det_denom = shared_info.tp_for_thresh + shared_info.fp_for_thresh;
	if (det_denom > 0)
	{
		cur_precision = (float)shared_info.tp_for_thresh / det_denom;
	}

	if (shared_info.unique_truth_count > 0)
	{
		cur_recall = (float)shared_info.tp_for_thresh / (float)shared_info.unique_truth_count;
	}

	if ((cur_precision + cur_recall) > 0.f)
	{
		f1_score = 2.f * cur_precision * cur_recall / (cur_precision + cur_recall);
	}

	*cfg_and_state.output
		<< ""						<< std::endl
		<< "-> for conf_thresh="	<< shared_info.thresh_calc_avg_iou
		<< ", precision="			<< cur_precision
		<< ", recall="				<< cur_recall
		<< ", F1 score="			<< f1_score
		<< ""						<< std::endl
		<< "-> for conf_thresh="	<< shared_info.thresh_calc_avg_iou
		<< ", TP="					<< shared_info.tp_for_thresh
		<< ", FP="					<< shared_info.fp_for_thresh
		<< ", FN="					<< shared_info.unique_truth_count - shared_info.tp_for_thresh
		<< ", average IoU="			<< shared_info.avg_iou * 100.0f << "%"
		<< ""						<< std::endl
		<< "-> IoU threshold="		<< shared_info.iou_threshold * 100.0f << "%, ";
	if (map_points)
	{
		*cfg_and_state.output << "used " << map_points << " recall points" << std::endl;
	}
	else
	{
		*cfg_and_state.output << "used area-under-curve for each unique recall" << std::endl;
	}

	mean_average_precision = (shared_info.number_of_classes > 0) ? (mean_average_precision / shared_info.number_of_classes) : 0.0;
	*cfg_and_state.output
		<< "-> mean average precision (mAP@" << std::setprecision(2) << iou_thresh << ")="
		<< Darknet::format_map_accuracy(mean_average_precision)
		<< std::endl;

	Darknet::update_f1_in_new_charts(-1, f1_score);

	const auto timestamp_end = std::chrono::high_resolution_clock::now();

	*cfg_and_state.output << "mAP calculations took a total of " << Darknet::format_duration_string(timestamp_end - timestamp_start, 1, Darknet::EFormatDuration::kTrim) << "." << std::endl;

	if (existing_net)
	{
		restore_network_recurrent_state(*existing_net);
	}
	else
	{
		free_network(shared_info.net);
	}

	return mean_average_precision;
}
