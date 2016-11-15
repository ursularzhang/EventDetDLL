// EventDetDll.h

#pragma once
#include <vector>
#include <deque>
#include <map>
#include "cv.h"
#include "highgui.h"
#include "CommonType.h"
#include "TrackList.h"
//#include "Miniball.hpp"
#include <windows.h>

using namespace std;
using namespace cv;

namespace EventDetection
{	
	typedef struct DensityData
	{
		string	ts;
		Point centre;		
		int count;
		vector<int> IDList;
	};

	typedef vector<DensityData> DensityHist;

	class EventDetDLL
	{
		// TODO: Add your methods for this class here.
	private:	
		string					prevTime;
		string					currTime;
		vector<Scalar>			colorList;
		int						width, height;
		Mat						currFrame;	
		string					dataFolder;

		//parameters from config file
		int						minGridSize;
		int						minGroupSize;
		Mat						eventMask;
		Mat						intrusionMapMask;
		int						congregationStayTime;
		int						congregationIgnoreTime;
		float					congregationSlope;
		float					groupingDistThreshold;
		int						dispersionCountThreshold;
		int						dispersionMinCount;
		float					dispersionSlope;
		int						reportEventMask;
		
		void		 			UpdateThread();
		void					StartThread();

		void					LoadConfig();

		void					LoadIntrusionSetting();
		bool					CheckIntrusion(float x, float y);
		bool					CheckIntrusion(double x, double y);

		void					CheckCandidateEvents();
		void					CalculateDensity();

		void					ClusterCurrentPointsbyDensity();
		bool					CheckSmallGathering(vector<RepPoint*> group, vector<int> &ids, BBOX &box);
		EventInfo				*lastCongregationEvent;
		EventInfo				*lastCandidate;
		
		void					CheckCongregation();
		void					CheckDispersion();
		EventInfo				*lastDispersionEvent;
		DensityHist				groupList;
		DensityHist				currGroupList;
		void					UpdateGroupList(vector<RepPoint*> group);
		void					RemoveOutdatedGroup();

	public:
		EventDetDLL();
		EventDetDLL(int procW, int procH, string folder);
		~EventDetDLL();
		TrackList				objects;
		vector<EventInfo*>		events;
		vector<EventInfo*>		candidateEvents;
		Mat						metaFrame;
		int						processBatchID;
		int						lastEventID;
		int						bufferSwitch;
		bool					isUpdating;
		bool					hasEvents;
		vector<RepPoint*>		PointBuffer;
		vector<RepPoint*>		PointBuffer_1;
		int						updateInterval;
		bool					saveOutput;

		//trung's interface:
		//void	GetLatestOutput(BBOXArr &curTrackInfo, int inWidth,  int inHeight);
		void AddPoints(BBOX box, int id, string ts, int frameID, int camID);
		void ClearOutdatedTracks();
		void UpdateStatus();
		void UpdatePoints(vector<RepPoint*> &buffer);
		void UpdateFrame(Mat img, string time, int count);
		void ReportEvents(vector<EventInfo*> &eventList);
		void UpdateFrame(Mat img, string time);
		void StopReporting(int mask);
	};
}
