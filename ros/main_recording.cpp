/*
 * Copyright Robert Bosch GmbH 2014
 * All rights reserved, also regarding any disposal, 
 * exploitation, reproduction, editing, distribution,
 * as well as in the event of application for industrial
 * property rights.
 *
 * Project   : TA
 * File      : main_recording.cpp
 * Created on: 18.03.2014
 * Author    : aum7si
 * 
 */

#include "ros/ros.h"
#include "std_msgs/String.h"
#include "rosbag/bag.h"
#include "rosbag/view.h"
#include "pcl_ros/point_cloud.h"
#include "pcl/point_types.h"
#include <pcl/point_cloud.h>
#include "Constants.h"
#include <iostream>
#include <string.h>
#include <stdio.h>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Pose.h>
#include <visualization_msgs/Marker.h>
#include <dirent.h>
#include <cmath>
#include <fstream>
#include "sensor_msgs/Image.h"
#include "sensor_msgs/image_encodings.h"

// NEEDS ROS FUERTE
 
int kbhit(void)
{
  struct termios oldt, newt;
  int ch;
  int oldf;
 
  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);
  oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
  fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);
 
  ch = getchar();
 
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  fcntl(STDIN_FILENO, F_SETFL, oldf);
 
  if(ch != EOF)
  {
    ungetc(ch, stdin);
    return 1;
  }
 
  return 0;
}

class rbros_Deepbag {
private:
	rosbag::Bag bag;
	bool bagging;
	int nr_samples;
	int total_nr_samples;
public:
	rbros_Deepbag() {
	    bagging = false;
	    total_nr_samples = 0;
	    nr_samples = 0;
	}
	~rbros_Deepbag() {}
	void openNew (std::string name) {
		if (!bagging) {
			bag.open(name, rosbag::bagmode::Write);
			bagging = true;
			nr_samples = 0;
		}
	}
	void close () {
		if (bagging) {
			bagging = false;
			bag.close();
			std::cout << "Recorded " << nr_samples << " samples\n";
			total_nr_samples += nr_samples;
			std::cout << "Recorded " << total_nr_samples << " samples in total\n";
		}
	}

	bool get_State(){
		return bagging;
	}

	int get_total_number_samples(){
		return total_nr_samples;
	}
	
	void cb_Position(const ros::MessageEvent<visualization_msgs::MarkerArray >& event, const std::string& topic) {
		if (bagging) {
			visualization_msgs::MarkerArray agents = *event.getMessage();
			std::string ns = "cluster_text";
			std::cout << "----------------------------------\n";
			std::cout << "[INFO] Number of Markers: " << agents.markers.size() << "\n";
			std::cout << "----------------------------------\n";

			/*for (int i=0; i < agents.markers.size(); i++){
				visualization_msgs::Marker agent = agents.markers[i];				
				if (!ns.compare(agent.ns)){
					geometry_msgs::Pose pose = agent.pose;
					std::cout << "----------------------------------\n";
					std::cout << "Person X-Coord: " << pose.position.x << "\n";
					std::cout << "Person Y-Coord: " << pose.position.y << "\n";
					std::cout << "Person Z-Coord: " << pose.position.z << "\n";
					std::cout << "----------------------------------\n";
				}
			}*/
			bag.write(topic.c_str(), ros::Time::now(), event.getMessage());
		}
	}

	void cb_QPCL(const ros::MessageEvent<pcl::PointCloud<USArrayPointType> >& event, const std::string& topic) {
		if (bagging) {
			/*std::cout << "----------------------------------\n";
			std::cout << "[INFO] Number of QPOints: " << event.getMessage()->width << "\n";
			std::cout << "----------------------------------\n";
			for (int i = 0; i < event.getMessage()->width; i++){
				std::cout << "emitterID: " << event.getMessage()->points[i].emitterID << "\n";
				std::cout << "arrayID: " << event.getMessage()->points[i].arrayID << "\n";
				std::cout << "x: " << event.getMessage()->points[i].x << "\n";
				std::cout << "y: " << event.getMessage()->points[i].y << "\n";
				std::cout << "z: " << event.getMessage()->points[i].z << "\n";
				std::cout << "----------------------------------\n";
			}*/
			bag.write(topic.c_str(), ros::Time::now(), event.getMessage());
			nr_samples++;
		}
	}

	void cb_Image(const ros::MessageEvent<sensor_msgs::Image >& event, const std::string& topic) {
		if (bagging) {
			bag.write(topic.c_str(), ros::Time::now(), event.getMessage());
		}
	}
	
	

	int generate_dataset(std::string path){

		/*------------------------------------MAKE A LIST OF ALL THE BAG FILES---------------------------------------------*/
		std::vector<std::string> bagfiles; 
		DIR *dir;
		struct dirent *ent;

		if ((dir = opendir (path.c_str())) != NULL) {
  			while ((ent = readdir (dir)) != NULL) {
				std::string file_name = ent->d_name;
				if (file_name.compare(".") && file_name.compare("..")){
					bagfiles.push_back(file_name);    					
				}
  			}
  			closedir (dir);
		} else {
  			perror ("");
  			return EXIT_FAILURE;
		}
		/*----------------------------------------------------------------------------------------------------------------*/
		
		/*------------------------------------LOAD ALL POINTCLOUDS AND DETECTED PERSONS-----------------------------------*/
		std::vector<pcl::PointCloud<USArrayPointType> > clouds;
		std::vector<ros::Time> cloud_times;
		std::vector<std::string> annotations;
		std::vector<visualization_msgs::MarkerArray> m_arrays;
		std::vector<ros::Time> ma_times;
		std::vector<std::string> topics;
		topics.push_back(std::string("/USArray_pc"));
		topics.push_back(std::string("/visualization_marker_array"));

		for(int i=0; i<bagfiles.size(); i++){
			std::string current_path(path);
			current_path.append(bagfiles[i]);
			std::string annotation = bagfiles[i].replace(bagfiles[i].end()-4,bagfiles[i].end(),"");
			std::cout << "Generating samples with annotation " << annotation << "\n";
			rosbag::Bag current_bag;
			current_bag.open(current_path,rosbag::bagmode::Read);
			rosbag::View view(current_bag, rosbag::TopicQuery(topics));

			BOOST_FOREACH(rosbag::MessageInstance const m, view){
				boost::shared_ptr<pcl::PointCloud<USArrayPointType> > pcl = m.instantiate<pcl::PointCloud<USArrayPointType> >();
				if (pcl != NULL){
					clouds.push_back(*pcl);
					cloud_times.push_back(m.getTime());
					annotations.push_back(annotation);
					/*std::cout << "emitterID: " << pcl->points[0].emitterID << "\n";
					std::cout << "arrayID: " << pcl->points[0].arrayID << "\n";
					std::cout << "time:" << m.getTime() << "\n";
					std::cout << "################################\n";
					sleep(4);*/
				}

				visualization_msgs::MarkerArray::ConstPtr ma = m.instantiate<visualization_msgs::MarkerArray>();
				if (ma != NULL){
					m_arrays.push_back(*ma);
					ma_times.push_back(m.getTime());
				}
			}
			current_bag.close();

			if (clouds.size() != cloud_times.size() || clouds.size() != annotations.size()) 
				std::cout << "[ERROR] LIST INTEGRITY VIOLATED\n";
			if (m_arrays.size() != ma_times.size()) 
				std::cout << "[ERROR] LIST INTEGRITY VIOLATED\n";
		}
		std::cout << "Found " << clouds.size() << " samples in total\n";
		std::cout << "Found " << cloud_times.size() << " sample times in total\n";
		std::cout << "Found " << annotations.size() << " annotations in total\n";		
		std::cout << "Found " << m_arrays.size() << " labels in total\n";
		std::cout << "Found " << ma_times.size() << " label times in total\n";

		/*----------------------------------------------------------------------------------------------------------------*/

		/*------------------------------------DETERMINING CORRESPONDING POSITION FOR EVERY SAMPLE-------------------------*/
		std::vector<visualization_msgs::MarkerArray*> ma_pointers;
		for(int i=0; i<cloud_times.size();i++){
			ros::Duration shortest_duration(60*60*24);
			int ma_ind;
			ros::Time cloud_time = cloud_times[i];
			for(int j=0; j<ma_times.size(); j++){
				ros::Time ma_time = ma_times[j];
				ros::Duration duration(cloud_time - ma_time);
				if(std::abs(duration.toSec()) < std::abs(shortest_duration.toSec())){
					shortest_duration = duration;
					ma_ind = j;
				}
				
			}	
			ma_pointers.push_back(&m_arrays[ma_ind]);
		}
		std::cout << "Have now " << ma_pointers.size() << " labels for "<< clouds.size() << " samples\n";
		
		/*----------------------------------------------------------------------------------------------------------------*/
		
		/*------------------------------------TRANSFORM POINTCLOUDS INTO LISTS OF POINTS----------------------------------*/
		std::vector<std::vector<USArrayPointType> > qpoint_lists;		

		for(int i=0; i<clouds.size(); i++){
			std::vector<USArrayPointType> qpoints;
			for(int j=0; j<clouds[i].points.size(); j++){
				qpoints.push_back(clouds[i].points[j]);
			}
			qpoint_lists.push_back(qpoints);
		}
		std::cout << "Have generated " << qpoint_lists.size() << " lists of QPoints.\n";
		
		/*----------------------------------------------------------------------------------------------------------------*/
		
		/*------------------------------------FIND THE ACTUAL POSITION IN THE MARKER ARRAYS-------------------------------*/
		std::vector<std::vector<double> > poses;
		std::string ns0 = "cluster_text";
		std::string ns1 = "cluster";
		int mult_poss = 0;
		for(int i=0; i<ma_pointers.size(); i++){
			int ma_array_length = ma_pointers[i]->markers.size();
			std::vector<visualization_msgs::Marker> poss_markers;
			bool found = false;

			for(int j=0; j<ma_array_length; j++){
				visualization_msgs::Marker marker = ma_pointers[i]->markers[j];				
				if (!ns0.compare(marker.ns) || !ns1.compare(marker.ns)){
					poss_markers.push_back(marker);
				}
			}
			
			for(int k=0; k<poss_markers.size();){
				if (poss_markers[k].pose.position.x > 4.4 || poss_markers[k].pose.position.y < -2.5 || poss_markers[k].pose.position.y > 2.5)
					poss_markers.erase(poss_markers.begin()+k);
				else
					k++;
			}
			
			if (poss_markers.size() == 1){
				found = true;
				std::vector<double> position;
				position.push_back(poss_markers[0].pose.position.x);
				position.push_back(poss_markers[0].pose.position.y);
				position.push_back(poss_markers[0].pose.position.z);
				poses.push_back(position);
			}
			
			if (poss_markers.size() > 1){
				int nr_active = 0;
				for (int k=0; k<poss_markers.size(); k++){
					if (!ns0.compare(poss_markers[k].ns))
						nr_active++;
				}
				if (nr_active > 1 || nr_active == 0){
					mult_poss++;
					std::vector<double> position;
					position.push_back(-2000);
					position.push_back(-2000);
					position.push_back(-2000);
					poses.push_back(position);
					found = true;
				} else {
					int ind_active = 0;
					for (int k=0; k<poss_markers.size(); k++){
						if (!ns0.compare(poss_markers[k].ns))
							ind_active = k;
							break;
					}
					std::vector<double> position;
					position.push_back(poss_markers[ind_active].pose.position.x);
					position.push_back(poss_markers[ind_active].pose.position.y);
					position.push_back(poss_markers[ind_active].pose.position.z);
					poses.push_back(position);
					found = true;
				}
			}
				
			if (ma_array_length == 0 || !found){
				std::vector<double> position;
				position.push_back(-1000);
				position.push_back(-1000);
				position.push_back(-1000);
				poses.push_back(position);
			}
		}
		std::cout << "Have found " << mult_poss << " samples to have multiple positions.\n";
		std::cout << "Have generated " << poses.size() << " poses.\n";
		
		/*----------------------------------------------------------------------------------------------------------------*/
		
		/*------------------------------------WRITE EVERYTHING TO THE FILES-----------------------------------------------*/
		std::ofstream samples;
		samples.open ("/home/mo/samples.txt");
		
		for(int i=0; i<qpoint_lists.size(); i++){
			for(int j=0; j<qpoint_lists[i].size(); j++){
  				samples << qpoint_lists[i][j].x << "," << qpoint_lists[i][j].y << "," << qpoint_lists[i][j].z << "," << qpoint_lists[i][j].plausibility << "," << qpoint_lists[i][j].power << "," << qpoint_lists[i][j].emitterID << "," << qpoint_lists[i][j].arrayID;
				if(j != qpoint_lists[i].size()-1)  samples << ":";  				
			}
			if(i != qpoint_lists.size()-1)	samples << ";";
		}
		samples.close();

		std::ofstream labels;
		labels.open ("/home/mo/labels.txt");
		for(int i=0; i<poses.size(); i++){
			labels << poses[i][0] << "," << poses[i][1] << "," << poses[i][2];			
			if(i != poses.size()-1)	labels << ";";
		}
		labels.close();

		std::ofstream annotation;
		annotation.open ("/home/mo/annotations.txt");
		for(int i=0; i<annotations.size(); i++){
  			annotation << annotations[i];			
			if(i != annotations.size()-1)	annotation << ";";
		}
		annotation.close();

		std::cout << "Done writing the dataset to the files.\n";

		/*----------------------------------------------------------------------------------------------------------------*/		
	} 
	



	int generate_dataset_integrated(std::string path){

		/*------------------------------------MAKE A LIST OF ALL THE BAG FILES---------------------------------------------*/
		std::vector<std::string> bagfiles; 
		DIR *dir;
		struct dirent *ent;

		if ((dir = opendir (path.c_str())) != NULL) {
  			while ((ent = readdir (dir)) != NULL) {
				std::string file_name = ent->d_name;
				if (file_name.compare(".") && file_name.compare("..")){
					bagfiles.push_back(file_name);    					
				}
  			}
  			closedir (dir);
		} else {
  			perror ("");
  			return EXIT_FAILURE;
		}
		/*----------------------------------------------------------------------------------------------------------------*/
		
		/*------------------------------------LOAD ALL POINTCLOUDS AND DETECTED PERSONS-----------------------------------*/
		std::vector<pcl::PointCloud<USArrayPointType> > clouds;
		std::vector<ros::Time> cloud_times;
		std::vector<std::string> annotations;
		std::vector<visualization_msgs::MarkerArray> m_arrays;
		std::vector<ros::Time> ma_times;
		std::vector<std::string> topics;
		topics.push_back(std::string("/USArray_pc"));
		topics.push_back(std::string("/visualization_marker_array"));

		for(int i=0; i<bagfiles.size(); i++){
			std::string current_path(path);
			current_path.append(bagfiles[i]);
			std::string annotation = bagfiles[i].replace(bagfiles[i].end()-4,bagfiles[i].end(),"");
			std::cout << "Generating samples with annotation " << annotation << "\n";
			rosbag::Bag current_bag;
			current_bag.open(current_path,rosbag::bagmode::Read);
			rosbag::View view(current_bag, rosbag::TopicQuery(topics));
			pcl::PointCloud<USArrayPointType> large_pcl;
			int cloud_count = 0;

			BOOST_FOREACH(rosbag::MessageInstance const m, view){
				boost::shared_ptr<pcl::PointCloud<USArrayPointType> > pcl = m.instantiate<pcl::PointCloud<USArrayPointType> >();
				if (pcl != NULL){
					for (int i = 0; i < pcl->width; i++){
						large_pcl.points.push_back(pcl->points[i]);
					}
					cloud_count++;
					if (cloud_count == 6){
						cloud_times.push_back(m.getTime());
						annotations.push_back(annotation);
						clouds.push_back(large_pcl);
						cloud_count = 0;
						//std::cout << "new aggregated point cloud built with length " << large_pcl.points.size() << "\n";					
						large_pcl.points.clear();

					}
				}

				visualization_msgs::MarkerArray::ConstPtr ma = m.instantiate<visualization_msgs::MarkerArray>();
				if (ma != NULL){
					m_arrays.push_back(*ma);
					ma_times.push_back(m.getTime());
				}

			}
			current_bag.close();

			if (clouds.size() != cloud_times.size() || clouds.size() != annotations.size()) 
				std::cout << "[ERROR] LIST INTEGRITY VIOLATED\n";
			if (m_arrays.size() != ma_times.size()) 
				std::cout << "[ERROR] LIST INTEGRITY VIOLATED\n";
		}
		std::cout << "Found " << clouds.size() << " samples in total\n";
		std::cout << "Found " << cloud_times.size() << " sample times in total\n";
		std::cout << "Found " << annotations.size() << " annotations in total\n";		
		std::cout << "Found " << m_arrays.size() << " labels in total\n";
		std::cout << "Found " << ma_times.size() << " label times in total\n";

		/*----------------------------------------------------------------------------------------------------------------*/

		/*------------------------------------DETERMINING CORRESPONDING POSITION FOR EVERY SAMPLE-------------------------*/
		std::vector<visualization_msgs::MarkerArray*> ma_pointers;
		for(int i=0; i<cloud_times.size();i++){
			ros::Duration shortest_duration(60*60*24);
			int ma_ind;
			ros::Time cloud_time = cloud_times[i];
			for(int j=0; j<ma_times.size(); j++){
				ros::Time ma_time = ma_times[j];
				ros::Duration duration(cloud_time - ma_time);
				if(std::abs(duration.toSec()) < std::abs(shortest_duration.toSec())){
					shortest_duration = duration;
					ma_ind = j;
				}
				
			}	
			ma_pointers.push_back(&m_arrays[ma_ind]);
		}
		std::cout << "Have now " << ma_pointers.size() << " labels for "<< clouds.size() << " samples\n";
		
		/*----------------------------------------------------------------------------------------------------------------*/
		
		/*------------------------------------TRANSFORM POINTCLOUDS INTO LISTS OF POINTS----------------------------------*/
		std::vector<std::vector<USArrayPointType> > qpoint_lists;		

		for(int i=0; i<clouds.size(); i++){
			std::vector<USArrayPointType> qpoints;
			for(int j=0; j<clouds[i].points.size(); j++){
				qpoints.push_back(clouds[i].points[j]);
			}
			qpoint_lists.push_back(qpoints);
		}
		std::cout << "Have generated " << qpoint_lists.size() << " lists of QPoints.\n";
		
		/*----------------------------------------------------------------------------------------------------------------*/
		
		/*------------------------------------FIND THE ACTUAL POSITION IN THE MARKER ARRAYS-------------------------------*/
		std::vector<std::vector<double> > poses;
		std::string ns0 = "cluster_text";
		std::string ns1 = "cluster";
		int mult_poss = 0;
		for(int i=0; i<ma_pointers.size(); i++){
			int ma_array_length = ma_pointers[i]->markers.size();
			std::vector<visualization_msgs::Marker> poss_markers;
			bool found = false;

			for(int j=0; j<ma_array_length; j++){
				visualization_msgs::Marker marker = ma_pointers[i]->markers[j];				
				if (!ns0.compare(marker.ns) || !ns1.compare(marker.ns)){
					poss_markers.push_back(marker);
				}
			}
			
			for(int k=0; k<poss_markers.size();){
				if (poss_markers[k].pose.position.x > 4.4 || poss_markers[k].pose.position.y < -2.5 || poss_markers[k].pose.position.y > 2.5)
					poss_markers.erase(poss_markers.begin()+k);
				else
					k++;
			}
			
			if (poss_markers.size() == 1){
				found = true;
				std::vector<double> position;
				position.push_back(poss_markers[0].pose.position.x);
				position.push_back(poss_markers[0].pose.position.y);
				position.push_back(poss_markers[0].pose.position.z);
				poses.push_back(position);
			}
			
			if (poss_markers.size() > 1){
				int nr_active = 0;
				for (int k=0; k<poss_markers.size(); k++){
					if (!ns0.compare(poss_markers[k].ns))
						nr_active++;
				}
				if (nr_active > 1 || nr_active == 0){
					mult_poss++;
					std::vector<double> position;
					position.push_back(-2000);
					position.push_back(-2000);
					position.push_back(-2000);
					poses.push_back(position);
					found = true;
				} else {
					int ind_active = 0;
					for (int k=0; k<poss_markers.size(); k++){
						if (!ns0.compare(poss_markers[k].ns))
							ind_active = k;
							break;
					}
					std::vector<double> position;
					position.push_back(poss_markers[ind_active].pose.position.x);
					position.push_back(poss_markers[ind_active].pose.position.y);
					position.push_back(poss_markers[ind_active].pose.position.z);
					poses.push_back(position);
					found = true;
				}
			}
				
			if (ma_array_length == 0 || !found){
				std::vector<double> position;
				position.push_back(-1000);
				position.push_back(-1000);
				position.push_back(-1000);
				poses.push_back(position);
			}
		}
		std::cout << "Have found " << mult_poss << " samples to have multiple positions.\n";
		std::cout << "Have generated " << poses.size() << " poses.\n";
		
		/*----------------------------------------------------------------------------------------------------------------*/
		
		/*------------------------------------WRITE EVERYTHING TO THE FILES-----------------------------------------------*/
		std::ofstream samples;
		samples.open ("/home/mo/samples_int.txt");
		
		for(int i=0; i<qpoint_lists.size(); i++){
			for(int j=0; j<qpoint_lists[i].size(); j++){
  				samples << qpoint_lists[i][j].x << "," << qpoint_lists[i][j].y << "," << qpoint_lists[i][j].z << "," << qpoint_lists[i][j].plausibility << "," << qpoint_lists[i][j].power << "," << qpoint_lists[i][j].emitterID << "," << qpoint_lists[i][j].arrayID;
				if(j != qpoint_lists[i].size()-1)  samples << ":";  				
			}
			if(i != qpoint_lists.size()-1)	samples << ";";
		}
		samples.close();

		std::ofstream labels;
		labels.open ("/home/mo/labels_int.txt");
		for(int i=0; i<poses.size(); i++){
			labels << poses[i][0] << "," << poses[i][1] << "," << poses[i][2];			
			if(i != poses.size()-1)	labels << ";";
		}
		labels.close();

		std::ofstream annotation;
		annotation.open ("/home/mo/annotations_int.txt");
		for(int i=0; i<annotations.size(); i++){
  			annotation << annotations[i];			
			if(i != annotations.size()-1)	annotation << ";";
		}
		annotation.close();

		std::cout << "Done writing the dataset to the files.\n";

		/*----------------------------------------------------------------------------------------------------------------*/		
	}
	
	
	int generate_dataset_integrated_ordered(std::string path){

			/*------------------------------------MAKE A LIST OF ALL THE BAG FILES---------------------------------------------*/
			std::vector<std::string> bagfiles; 
			DIR *dir;
			struct dirent *ent;

			if ((dir = opendir (path.c_str())) != NULL) {
	  			while ((ent = readdir (dir)) != NULL) {
					std::string file_name = ent->d_name;
					if (file_name.compare(".") && file_name.compare("..")){
						bagfiles.push_back(file_name);    					
					}
	  			}
	  			closedir (dir);
			} else {
	  			perror ("");
	  			return EXIT_FAILURE;
			}
			/*----------------------------------------------------------------------------------------------------------------*/
			
			/*------------------------------------LOAD ALL POINTCLOUDS AND DETECTED PERSONS-----------------------------------*/
			std::vector<pcl::PointCloud<USArrayPointType> > clouds;
			std::vector<ros::Time> cloud_times;
			std::vector<std::string> annotations;
			std::vector<visualization_msgs::MarkerArray> m_arrays;
			std::vector<ros::Time> ma_times;
			std::vector<std::string> topics;
			topics.push_back(std::string("/USArray_pc"));
			topics.push_back(std::string("/visualization_marker_array"));

			for(int i=0; i<bagfiles.size(); i++){
				std::string current_path(path);
				current_path.append(bagfiles[i]);
				std::string annotation = bagfiles[i].replace(bagfiles[i].end()-4,bagfiles[i].end(),"");
				std::cout << "Generating samples with annotation " << annotation << "\n";
				rosbag::Bag current_bag;
				current_bag.open(current_path,rosbag::bagmode::Read);
				rosbag::View view(current_bag, rosbag::TopicQuery(topics));
				pcl::PointCloud<USArrayPointType> large_pcl;
				int cloud_count = 0;

				BOOST_FOREACH(rosbag::MessageInstance const m, view){
					boost::shared_ptr<pcl::PointCloud<USArrayPointType> > pcl = m.instantiate<pcl::PointCloud<USArrayPointType> >();
					if (pcl != NULL){
						for (int i = 0; i < pcl->width; i++){
							large_pcl.points.push_back(pcl->points[i]);
						}
						cloud_count++;
						if (cloud_count == 6){
							cloud_times.push_back(m.getTime());
							annotations.push_back(annotation);
							clouds.push_back(large_pcl);
							cloud_count = 0;
							//std::cout << "new aggregated point cloud built with length " << large_pcl.points.size() << "\n";					
							large_pcl.points.clear();

						}
					}

					visualization_msgs::MarkerArray::ConstPtr ma = m.instantiate<visualization_msgs::MarkerArray>();
					if (ma != NULL){
						m_arrays.push_back(*ma);
						ma_times.push_back(m.getTime());
					}

				}
				current_bag.close();

				if (clouds.size() != cloud_times.size() || clouds.size() != annotations.size()) 
					std::cout << "[ERROR] LIST INTEGRITY VIOLATED\n";
				if (m_arrays.size() != ma_times.size()) 
					std::cout << "[ERROR] LIST INTEGRITY VIOLATED\n";
			}
			std::cout << "Found " << clouds.size() << " samples in total\n";
			std::cout << "Found " << cloud_times.size() << " sample times in total\n";
			std::cout << "Found " << annotations.size() << " annotations in total\n";		
			std::cout << "Found " << m_arrays.size() << " labels in total\n";
			std::cout << "Found " << ma_times.size() << " label times in total\n";

			/*----------------------------------------------------------------------------------------------------------------*/

			/*------------------------------------DETERMINING CORRESPONDING POSITION FOR EVERY SAMPLE-------------------------*/
			std::vector<visualization_msgs::MarkerArray*> ma_pointers;
			for(int i=0; i<cloud_times.size();i++){
				ros::Duration shortest_duration(60*60*24);
				int ma_ind;
				ros::Time cloud_time = cloud_times[i];
				for(int j=0; j<ma_times.size(); j++){
					ros::Time ma_time = ma_times[j];
					ros::Duration duration(cloud_time - ma_time);
					if(std::abs(duration.toSec()) < std::abs(shortest_duration.toSec())){
						shortest_duration = duration;
						ma_ind = j;
					}
					
				}	
				ma_pointers.push_back(&m_arrays[ma_ind]);
			}
			std::cout << "Have now " << ma_pointers.size() << " labels for "<< clouds.size() << " samples\n";
			
			/*----------------------------------------------------------------------------------------------------------------*/
			
			/*------------------------------------TRANSFORM POINTCLOUDS INTO LISTS OF POINTS----------------------------------*/
			std::vector<std::vector<USArrayPointType> > qpoint_lists;		

			for(int i=0; i<clouds.size(); i++){
				std::vector<USArrayPointType> qpoints;
				for(int j=0; j<clouds[i].points.size(); j++){
					qpoints.push_back(clouds[i].points[j]);
				}
				qpoint_lists.push_back(qpoints);
			}
			std::cout << "Have generated " << qpoint_lists.size() << " lists of QPoints.\n";
			
			/*----------------------------------------------------------------------------------------------------------------*/
			
			/*------------------------------------FIND THE ACTUAL POSITION IN THE MARKER ARRAYS-------------------------------*/
			std::vector<std::vector<double> > poses;
			std::string ns0 = "cluster_text";
			std::string ns1 = "cluster";
			int mult_poss = 0;
			for(int i=0; i<ma_pointers.size(); i++){
				int ma_array_length = ma_pointers[i]->markers.size();
				std::vector<visualization_msgs::Marker> poss_markers;
				bool found = false;

				for(int j=0; j<ma_array_length; j++){
					visualization_msgs::Marker marker = ma_pointers[i]->markers[j];				
					if (!ns0.compare(marker.ns) || !ns1.compare(marker.ns)){
						poss_markers.push_back(marker);
					}
				}
				
				for(int k=0; k<poss_markers.size();){
					if (poss_markers[k].pose.position.x > 4.4 || poss_markers[k].pose.position.y < -2.5 || poss_markers[k].pose.position.y > 2.5)
						poss_markers.erase(poss_markers.begin()+k);
					else
						k++;
				}
				
				if (poss_markers.size() == 1){
					found = true;
					std::vector<double> position;
					position.push_back(poss_markers[0].pose.position.x);
					position.push_back(poss_markers[0].pose.position.y);
					position.push_back(poss_markers[0].pose.position.z);
					poses.push_back(position);
				}
				
				if (poss_markers.size() > 1){
					int nr_active = 0;
					for (int k=0; k<poss_markers.size(); k++){
						if (!ns0.compare(poss_markers[k].ns))
							nr_active++;
					}
					if (nr_active > 1 || nr_active == 0){
						mult_poss++;
						std::vector<double> position;
						position.push_back(-2000);
						position.push_back(-2000);
						position.push_back(-2000);
						poses.push_back(position);
						found = true;
					} else {
						int ind_active = 0;
						for (int k=0; k<poss_markers.size(); k++){
							if (!ns0.compare(poss_markers[k].ns))
								ind_active = k;
								break;
						}
						std::vector<double> position;
						position.push_back(poss_markers[ind_active].pose.position.x);
						position.push_back(poss_markers[ind_active].pose.position.y);
						position.push_back(poss_markers[ind_active].pose.position.z);
						poses.push_back(position);
						found = true;
					}
				}
					
				if (ma_array_length == 0 || !found){
					std::vector<double> position;
					position.push_back(-1000);
					position.push_back(-1000);
					position.push_back(-1000);
					poses.push_back(position);
				}
			}
			std::cout << "Have found " << mult_poss << " samples to have multiple positions.\n";
			std::cout << "Have generated " << poses.size() << " poses.\n";
			
			/*----------------------------------------------------------------------------------------------------------------*/

			/*------------------------------------ORDER ALL SAMPLES BY TIMESTAMPS (INC)---------------------------------------*/
			std::vector<USArrayPointType> q_swap;
			ros::Time t_swap;
			std::string a_swap;
			std::vector<double> p_swap;
			ros::Time earliest;
			int earliest_ind;
			
			for (int i=0; i<qpoint_lists.size(); i++){
				earliest = cloud_times[i];
				earliest_ind = i;
						
				for (int j=i+1; j<qpoint_lists.size(); j++){
					if (cloud_times[j]  < earliest){
						earliest_ind = j;
						earliest = cloud_times[j];
					}
				}
				
				q_swap = qpoint_lists[i];
				qpoint_lists[i] = qpoint_lists[earliest_ind];
				qpoint_lists[earliest_ind] = q_swap;
				t_swap = cloud_times[i];
				cloud_times[i] = cloud_times[earliest_ind];
				cloud_times[earliest_ind] = t_swap;
				a_swap = annotations[i];
				annotations[i] = annotations[earliest_ind];
				annotations[earliest_ind] = a_swap;
				p_swap = poses[i];
				poses[i] = poses[earliest_ind];
				poses[earliest_ind] = p_swap;
			}
			
			for (int j=0; j<cloud_times.size(); j++){
				std::cout << cloud_times[j] << "\n";
			}
			/*----------------------------------------------------------------------------------------------------------------*/
			
			/*------------------------------------WRITE EVERYTHING TO THE FILES-----------------------------------------------*/
			std::ofstream samples;
			samples.open ("/home/mo/samples_int_ordered.txt");
			
			for(int i=0; i<qpoint_lists.size(); i++){
				for(int j=0; j<qpoint_lists[i].size(); j++){
	  				samples << qpoint_lists[i][j].x << "," << qpoint_lists[i][j].y << "," << qpoint_lists[i][j].z << "," << qpoint_lists[i][j].plausibility << "," << qpoint_lists[i][j].power << "," << qpoint_lists[i][j].emitterID << "," << qpoint_lists[i][j].arrayID;
					if(j != qpoint_lists[i].size()-1)  samples << ":";  				
				}
				if(i != qpoint_lists.size()-1)	samples << ";";
			}
			samples.close();

			std::ofstream labels;
			labels.open ("/home/mo/labels_int_ordered.txt");
			for(int i=0; i<poses.size(); i++){
				labels << poses[i][0] << "," << poses[i][1] << "," << poses[i][2];			
				if(i != poses.size()-1)	labels << ";";
			}
			labels.close();

			std::ofstream annotation;
			annotation.open ("/home/mo/annotations_int_ordered.txt");
			for(int i=0; i<annotations.size(); i++){
	  			annotation << annotations[i];			
				if(i != annotations.size()-1)	annotation << ";";
			}
			annotation.close();

			std::cout << "Done writing the dataset to the files.\n";

			/*----------------------------------------------------------------------------------------------------------------*/		
		}


	std::string time2str(double ros_t) {
	  char buf[1024]      = "";
	  time_t t = ros_t;
	  struct tm *tms = localtime(&t);
	  strftime(buf, 1024, "%Y-%m-%d-%H-%M-%S", tms);
	  return std::string(buf);
	}
};







int main(int argc, char** argv) {
  ros::init(argc, argv, "RBROS_DeepROSBag");
  ros::NodeHandle nh;
  std::string imageTopic = "/image_raw";
  std::string qPointTopic = "/USArray_pc";
  std::string positionTopic = "/visualization_marker_array";
  std::string path = "/home/mo/DeepSamples/";

  /// instance of bagging class
  rbros_Deepbag bag;
  //tf::TransformListener listener;
  
  
  std::cout << "A simple node used to record samples for the \"Deep Learning for Person Classification\" master's thesis.\n\n";
  //std::cout << "Please enter the topic name for the Q-Points: ";  
  //std::getline(std::cin, qPointTopic);
  //std::cout << "You chose the topic to be: " << qPointTopic << "\n\n";
  //std::cout << "Please enter the path of the directory in which to store the bag files: ";
  //std::getline(std::cin, path);
  //std::cout << "You chose the path to be: " << path << "\n\n";
  std::cout << "[USAGE] 'r' to start recording, 's' to stop recording, 'd' to generate the dataset, 'x' to close the program.\n\n";


  // all subscriptions to be bagged
 
  ros::Subscriber sub_Position  = nh.subscribe<visualization_msgs::MarkerArray>(positionTopic, 100, boost::bind(&rbros_Deepbag::cb_Position, &bag, _1, positionTopic));

  ros::Subscriber sub_QPoints  = nh.subscribe<pcl::PointCloud<USArrayPointType> >(qPointTopic, 100, boost::bind(&rbros_Deepbag::cb_QPCL, &bag, _1, qPointTopic));

ros::Subscriber sub_Images  = nh.subscribe<sensor_msgs::Image>(imageTopic, 100, boost::bind(&rbros_Deepbag::cb_Image, &bag, _1, imageTopic));
  
  /// ROS looping parameters.
  ros::Rate loop_rate(10);
  ros::Time ROSnow;
  ros::Time ROSstart = ros::Time::now();


  
  while (ros::ok()) {
	ros::spinOnce();

	char command = '0';
	if(kbhit()){
		command = getc(stdin);
		std::cout << "\n";
		if (bag.get_State() && command == 's') {
			std::cout << "\nStopping the recording.\n";		
			bag.close();
		}else if (!bag.get_State() && command == 'r'){
				std::string annotation;
				std::cout << "Please enter the annotation of the recorded samples: ";  
  				std::getline(std::cin, annotation);
				std::string Name = path;
				//Name.append(bag.time2str(ros::Time::now().toSec()));
				Name.append(annotation);				
				Name.append(".bag");
				std::cout << "\nStarting to record to " << Name << "\n";
				bag.openNew(Name);
			
		}else if (command == 'x'){
			if (bag.get_State()){
				std::cout << "\nStopping the recording before quitting.\n";
				bag.close();
			}
			std::cout << "\nQuitting...\n";
			return 0;

		}else if (command == 'd'){
			bag.generate_dataset_integrated_ordered(path.c_str());
		}
	}
	loop_rate.sleep();	
  }
  return 0;
}



