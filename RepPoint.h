#pragma once
#include "Stdafx.h"
#include <vector>
#include <deque>
#include "EventType.h"
#include "CommonType.h"
#include <fstream>


using namespace std;

namespace EventDetection
{
	const float					DirectionThre = 45.0;
	const int					IntrusionListSize = 30*12;

	typedef struct EventInfo
	{
		int			type;	//bitwise event types
		string		startTime;
		string		endTime;
		vector<int>	objIDs;		//ids of objects involved
		bool		reported;
		bool		isCandidate;
		BBOX		box;
	};

	class Point
	{
	public:
		double x; 
		double y;

	public:
		Point(){};
		Point(float _x, float _y) {x = _x; y = _y;}
		~Point(){};

		Point& operator=(const Point& other) // copy assignment
		{
			if (this != &other) 
			{ // self-assignment check expected
				this->x = other.x;
				this->y = other.y;
			}
			return *this;
		}
	};

	class Line 
	{
	public:
		double _slope, _yInt;

	public:
		double getYforX(double x) 
		{
			return _slope*x + _yInt;
		}

		// Construct line from points
		bool fitPoints(const std::vector<Point> &pts) 
		{
			int nPoints = pts.size();
			if( nPoints < 2 ) 
			{
				// Fail: infinitely many lines passing through this single point
				return false;
			}
			double sumX=0, sumY=0, sumXY=0, sumX2=0;
			for(int i=0; i<nPoints; i++) 
			{
				sumX += pts[i].x;
				sumY += pts[i].y;
				sumXY += pts[i].x * pts[i].y;
				sumX2 += pts[i].x * pts[i].x;
			}

			double xMean = sumX / nPoints;
			double yMean = sumY / nPoints;
			double denominator = sumX2 - sumX * xMean;
			// You can tune the eps (1e-7) below for your specific task
			if( std::fabs(denominator) < 1e-7 ) 
			{
				// Fail: it seems a vertical line
				return false;
			}

			_slope = (sumXY - sumX * yMean) / denominator;
			_yInt = yMean - _slope * xMean;
			return true;
		}
	};

	class RepPoint
	{
	private:
		RepPoint* prevPoint;

	public:
		BBOX box;
		Point pos;
		string ts;
		int frameID;
		int camID;
		int objID;

	public:
		RepPoint(BBOX _box, string _ts, int _frameID, int _camID, int _objID);
		RepPoint(double x, double y);
		~RepPoint();
		RepPoint* GetPrevPoint();
		RepPoint* GetPrevPointbySec(int sec);
		void SetPrevPoint(RepPoint* point);
	};





	class RepVector
	{
	private: 
		Point start;
		Point end;
		double componentX;
		double componentY;

	public:
		RepVector(){};
		RepVector(Point s, Point e);
		~RepVector(){};

		static double DotProduct(RepVector v1, RepVector v2);
		static double Length(RepVector v);
	};

	class RepTrace
	{
	private:
		RepPoint*	root;
		//vector<RepTrace*> events;
		vector<RepPoint*> RPList;
		deque<int> intrusionStatus;
		int intrusionCount;
		float CalcalatePast10Speed();
		string		lastTime;

	public:
		RepTrace();
		~RepTrace();
		int ID;
		int length;
		float curSpeed;
		float curAcceleration;
		float maxAcceleration;
		float avgSpeed;
		float avgSpeed_10; //average speed for the past 10 seconds
		float distance;
		float duration; // in seconds
		bool updated;
		bool outdated;
		bool loiterReported;
		int eventStatus;
		int eventToggle;
		int directionChangeCount;
		static int loiterStayTimeThreshold;
		static int outdatedTimeThreshold;
		static float runningSpeedThreshold;
		static float runningAccelerationThreshold;

		static float timeDiff(string ts1, string ts2);
		static string timeAddSec(string ts1, float sec);
		static float GetDistance(RepPoint* p1, RepPoint* p2);
		static float GetDuration(RepPoint* p1, RepPoint* p2);
		void UpdateStatus();
		RepPoint* GetLastRP();
		RepPoint* GetFirstRP();
		void SetRoot(RepPoint* _point);
		RepPoint* GetRoot();
		void AppendPoint(RepPoint* _root);
		RepPoint* GetFirstPoint();
		int CheckTrackLen();
		void CheckOutdateStatus(string ts);
		void UpdateLastTime(string ts);
		void InsertIntrusionStatus(bool intruded);
	};

}