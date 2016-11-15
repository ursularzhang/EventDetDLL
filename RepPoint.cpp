#include "Stdafx.h"
#include "RepPoint.h"
#include <time.h>
#include <iostream>



#include <log4cpp/Portability.hh>
#include <log4cpp/Category.hh>
#include <log4cpp/PropertyConfigurator.hh>
#pragma comment(lib, "Ws2_32.lib")


using namespace std;
using namespace EventDetection;
//#define DEBUG

int RepTrace::loiterStayTimeThreshold;
int RepTrace::outdatedTimeThreshold;
float RepTrace::runningSpeedThreshold;
float RepTrace::runningAccelerationThreshold;

RepPoint::RepPoint(double _x, double _y)
{
	pos.x = _x; pos.y = _y;
}
RepPoint::RepPoint(BBOX _box, string _ts, int _frameID, int _camID, int _objID)
{
	prevPoint = nullptr;
	box.bboxL = _box.bboxL;	box.bboxT = _box.bboxT;
	box.bboxW = _box.bboxW; box.bboxH = _box.bboxH;
	pos.x = box.bboxL + box.bboxW/2.0;
	pos.y = box.bboxT + box.bboxH/2.0;
	ts = _ts;
	frameID = _frameID;
	camID = _camID;
	objID = _objID;
}

RepPoint::~RepPoint()
{
	if (prevPoint != nullptr) 
		delete prevPoint;
}

void RepPoint::SetPrevPoint(RepPoint* point)
{
	prevPoint = point;
}

RepPoint* RepPoint::GetPrevPoint()
{
	return prevPoint;
}

RepPoint* RepPoint::GetPrevPointbySec(int sec)
{
	RepPoint* pt = GetPrevPoint();
	while (RepTrace::timeDiff(ts, pt->ts) < sec)
	{
		if (pt == nullptr)
			break;
		pt = pt->GetPrevPoint();
	}
	return pt;
}

RepVector::RepVector(EventDetection::Point s, EventDetection::Point e)
{
	start = s; 
	end = e;
	componentX = e.x - s.x;
	componentY = e.y - s.y;
}

double RepVector::DotProduct(RepVector v1, RepVector v2)
{
	return v1.componentX * v2.componentX + v1.componentY * v2.componentY;
}

double RepVector::Length(RepVector v)
{
	return sqrt(pow(v.componentX, 2) + pow(v.componentY, 2));
}

RepTrace::RepTrace()
{
	length = 0;
	intrusionCount = 0;
	ID = -1;
	outdated = false;
	updated = true;
	loiterReported = false;
	curSpeed = 0.0; avgSpeed = 0.0;
	distance = 0.0; duration = 0.0;
	curAcceleration = 0.0; avgSpeed_10 = 0.0;
	directionChangeCount = 0;
	root = nullptr;
	eventStatus = 0;
	eventToggle = 0;
	maxAcceleration = 0.0;
}

RepTrace::~RepTrace()
{
	delete root;
}

RepPoint* RepTrace::GetRoot()
{
	//cout << root->objID << endl;
	return root;
}

RepPoint* RepTrace::GetFirstRP()
{
	return RPList.front();
}

RepPoint* RepTrace::GetLastRP()
{
	return RPList.back();
}

void RepTrace::SetRoot(RepPoint* _point)
{
	_point->SetPrevPoint(root);
	root = _point;
	//length = CheckTrackLen();
}

void RepTrace::AppendPoint(RepPoint* _point)
{
	SetRoot(_point);
	this->lastTime = _point->ts;
	length ++;
	this->updated = false;
#ifdef DEBUG
	string fname = ".\\eventOutput\\track" + to_string(this->GetRoot()->objID) + ".txt";
	FILE* f = fopen(fname.c_str(), "a");
	fprintf(f, "\n%s addPoint\n", _point->ts.c_str());
	fprintf(f, "Length: %d\n", length);
	RepPoint* p = this->GetRoot();
	while (p != nullptr)
	{
		fprintf(f, "\t%s %.3f %.3f\n", p->ts.c_str(), p->pos.x, p->pos.y);
		p = p->GetPrevPoint();
	}
	fclose(f);
#endif

}

void RepTrace::CheckOutdateStatus(string ts)
{
	float offTime = RepTrace::timeDiff(ts, lastTime);
	if (offTime > RepTrace::outdatedTimeThreshold)
		outdated = true;
}

void RepTrace::UpdateLastTime(string ts)
{
	lastTime = ts;
}

void RepTrace::UpdateStatus()
{
	if (updated) 
		return;
	//update the representative points
	if (length > 1)
	{
		//update the speed, distance, duration
		RepPoint* prev = root->GetPrevPoint();
		float dist = RepTrace::GetDistance(root, prev);
		//cout << root->objID << " " << length << " "  << dist << endl;
		float secs = RepTrace::GetDuration(root, prev);
		float prevSpeed = curSpeed;
		curSpeed = dist / secs;
		curAcceleration = (curSpeed - prevSpeed) / secs;
		if (curAcceleration > maxAcceleration)
			maxAcceleration = curAcceleration;
		distance += dist;
		duration = RepTrace::GetDuration(root, GetFirstPoint());
		avgSpeed = distance / duration;
		avgSpeed_10 = CalcalatePast10Speed();

		//update loiter status
		if (!loiterReported && EventType::GetBit(eventStatus, loiterMask) != 1)
		{
			if (duration > loiterStayTimeThreshold) 
				EventType::SetBit(eventStatus, loiterMask);
		}

		//update abnormal loitering status
		if (RPList.size() > 1)
		{
			//update direction trajectory
			RepVector v1(root->pos, prev->pos);
			RepVector v2(prev->pos, prev->GetPrevPoint()->pos); 
			double dot = RepVector::DotProduct(v1, v2);
			double len1 = RepVector::Length(v1);
			double len2 = RepVector::Length(v2);
			double cosAngle = dot / (len1 * len2);
			double angle = acos(cosAngle);
			//if (angle > PI / 2)
			if (angle > 3.14159265358979323846/2)
			{
				RPList.push_back(root);
				directionChangeCount ++;
			}
		}
		else
			RPList.push_back(root);

		//update intrusion status
		if ((float)intrusionCount / IntrusionListSize > 0.5)
			EventType::SetBit(eventStatus, intrusionMask);
		else 
			EventType::ClearBit(eventStatus, intrusionMask);
	}
	else
	{
		RPList.push_back(root);
	}

	updated = true;
}

void RepTrace::InsertIntrusionStatus(bool intruded)
{
	if (intruded)
	{
		intrusionStatus.push_back(1);
		intrusionCount ++;
	}
	else
		intrusionStatus.push_back(0);

	if (intrusionStatus.size() > IntrusionListSize)
	{
		if (intrusionStatus.front() == 1)
			intrusionCount --;
		intrusionStatus.pop_front();
	}
}
	
float RepTrace::timeDiff(string time1, string time2)
{
#ifdef DEBUG
	ofstream f;
	f.open(".\\eventOutput\\timeDiff.txt");
#endif
	int y1, M1, d1, H1, m1, s1, f1;
	sscanf(time1.c_str(), "%04d%02d%02d%02d%02d%02d.%03d", &y1, &M1, &d1, &H1, &m1, &s1, &f1);
	int y2, M2, d2, H2, m2, s2, f2;
	sscanf(time2.c_str(), "%04d%02d%02d%02d%02d%02d.%03d", &y2, &M2, &d2, &H2, &m2, &s2, &f2);

	if ((y1 != y2) || (M1 != M2) || (d1 != d2)) 
		return 0.0;

	float sec1 = (H1 * 60 + m1 ) * 60 + s1 + f1/1000.0;
	float sec2 = (H2 * 60 + m2 ) * 60 + s2 + f2/1000.0;
	//cout << time1.c_str() << " " << time2.c_str() << endl;
	//cout << H1 << " " << m1 << " " << s1 << " " << f1 << endl;
	//cout << H2 << " " << m2 << " " << s2 << " " << f2 << endl;
	//cout << sec1 << " " << sec2 << endl;
	//cout << sec1 - sec2 << endl;
	return sec1 - sec2;
}

string RepTrace::timeAddSec(string time1, float sec)
{
	int y1, M1, d1, H1, m1, s1, f1;
	sscanf(time1.c_str(), "%04d%02d%02d%02d%02d%02d.%03d", &y1, &M1, &d1, &H1, &m1, &s1, &f1);
	int H2 = H1, m2 = m1, s2, f2 = f1;
	int allsec = (H1 * 60 + m1) * 60 + s1 + sec;
	H2 = (allsec / 60) / 60;
	m2 = (allsec / 60) % 60; 
	s2 = allsec % 60;
	//if (s2 >= 60 || s2 < 0)
	//{
	//	m2 += s2 / 60;
	//	s2 = s2 % 60;
	//}
	//if (m2 >= 60 || m2 < 0)
	//{
	//	H2 += m2 / 60;
	//	m2 = m2 % 60;
	//}

	char ts[30];
	sprintf(ts, "%04d%02d%02d%02d%02d%02d.%03d", y1, M1, d1, H2, m2, s2, f2);
	return (string(ts));
}


float RepTrace::GetDistance(RepPoint* p1, RepPoint* p2)
{
#ifdef DEBUG
	ofstream file;
	file.open(".\\eventOutput\\getduration.txt", ios::app);
	file << p1->objID << " " << p1->pos.x << " " << p1->pos.y << endl;
	file << p2->objID << " " << p2->pos.x << " " << p2->pos.y << endl;
	file.close();	
#endif
	return sqrt(pow(p1->pos.x - p2->pos.x, 2) + pow(p1->pos.y - p2->pos.y, 2));
}

float RepTrace::GetDuration(RepPoint* p1, RepPoint* p2)
{
#ifdef DEBUG
	ofstream file;
	file.open(".\\eventOutput\\getduration.txt", ios::app);
	file << p1->objID << " " << p1->ts << " " << p2->ts << endl;
	file.close();
#endif
	return abs(timeDiff(p1->ts, p2->ts));
}

float RepTrace::CalcalatePast10Speed()
{

	float speed = 0.0;
	RepPoint *obj = root, *prevObj = root->GetPrevPoint();
	float duration = RepTrace::GetDuration(obj, prevObj);
	float dist = RepTrace::GetDistance(obj, prevObj);

#ifdef DEBUG
	ofstream f;
	if (root->objID == 6)
	{
		f.open(".\\eventOutput\\calSpeed.txt", ofstream::app);
		f << root->objID << " " << length <<  endl;
		f << "Duration: " << duration << endl;;
		f << "Dist: " << dist << endl;
	}
#endif

	while (duration < 10.0)
	{
		obj = prevObj;
		prevObj = obj->GetPrevPoint();
		if (prevObj == nullptr)
			break;
		duration += RepTrace::GetDuration(obj, prevObj);
		dist += RepTrace::GetDistance(obj, prevObj);
#ifdef DEBUG
		if (root->objID == 6)
		{
			f << obj->pos.x << " " << obj->pos.y << "->" << prevObj->pos.x << " " << prevObj->pos.y << endl;
			f << "Duration: " << duration << endl;;
			f << "Dist: " << dist << endl;
		}
#endif
	}
	speed = dist / duration;

#ifdef DEBUG
	if (root->objID == 6)
	{
		f << "speed: " << speed << endl;
		f.close();
	}
#endif
	return speed;
}

int RepTrace::CheckTrackLen()
{
	int len = 1;
	RepPoint* obj = root->GetPrevPoint();
	while (obj != nullptr)
	{
		len++;
		obj = obj->GetPrevPoint();
	}
	return len;		
}

RepPoint* RepTrace::GetFirstPoint()
{
	RepPoint* obj = root->GetPrevPoint();
	RepPoint* firstObj = root;
	while (obj != nullptr)
	{
		firstObj = obj;
		obj = obj->GetPrevPoint();
	}
	return firstObj;
}	
