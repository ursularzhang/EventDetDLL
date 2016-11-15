// This is the main DLL file.

#include "stdafx.h"
#include "EventDetDll.h"
#include <time.h>
#include <utility>
#include <iostream>
#include <fstream>

//for log4cpp
#include <log4cpp/Portability.hh>
#include <log4cpp/Category.hh>
#include <log4cpp/PropertyConfigurator.hh>
#pragma comment(lib, "Ws2_32.lib")


using namespace cv;
using namespace EventDetection;

//#define DEBUG
#define DENSITYGRIDNUM 8
EventDetDLL* pthis;

int						maxDensityR, maxDensityC;
DensityData				currDensity[DENSITYGRIDNUM][DENSITYGRIDNUM];
DensityHist				densMap[DENSITYGRIDNUM][DENSITYGRIDNUM];
DensityData				maxDensity[DENSITYGRIDNUM][DENSITYGRIDNUM];

float OverlappedRatio(BBOX box0, BBOX box1)
{
	float ratio;
	int x0 = box0.bboxL, y0 = box0.bboxT;
	int w0 = box0.bboxW, h0 = box0.bboxH;
	int x1 = box1.bboxL, y1 = box1.bboxT;
	int w1 = box1.bboxW, h1 = box1.bboxH;

	if ((x0 < x1 && x0+w0 < x1) || (x0 > x1 && x0 > x1+w1) 
		|| (y0 < y1 && y0+h0 < y1) || (y0 > y1 && y0 > y1+h1))
		return 2.0;

	int overlap_x0, overlap_x1, overlap_y0, overlap_y1;
	overlap_x0 = (x0 < x1) ? x1 : x0;
	overlap_x1 = ((x0+w0) < (x1+w1)) ? x0+w0 : x1+w1;

	overlap_y0 = (y0 < y1) ? y1 : y0;
	overlap_y1 = ((y0+h0) < (y1+h1)) ? y0+h0 : y1+h1;

	ratio = (float)(overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0) / (float)(w0 * h0);

	return ratio;	
}

void EventDetDLL::ClearOutdatedTracks()
{

	for (int i = 0; i < objects.length; i++)
	{
		RepTrace *trace = objects.GetTrack(i);
		if (trace->outdated)
		{

			objects.DeleteTrack(i);
		}
	}

}


cv::Scalar GetDrawColor(int objId)
{
	int nMax = 16;
	int colorIdx = objId % nMax;
	Scalar	p;

	switch(colorIdx)
	{
	 case 0:
		 p = Scalar(255,0,0);
		 break;
	 case 1:
		 p = Scalar(0,255,0);
		 break;
	 case 2:
		 p = Scalar(0,0,255);
		 break;
	 case 3:
		 p = Scalar(255,255,0);
		 break;
	 case 4:
		 p = Scalar(255,0,255);
		 break;
	 case 5:
		 p = Scalar(0,255,255);
		 break;
	 case 6:
		 p = Scalar(255,255,255);
		 break;
	 case 7:
		 p = Scalar(128,0,128);
		 break;
	 case 8:
		 p = Scalar(128,128,0);
		 break;
	 case 9:
		 p = Scalar(128,128,128);
		 break;
	 case 10:
		 p = Scalar(255,128,0);
		 break;
	 case 11:
		 p = Scalar(0,128,128);
		 break;
	 case 12:
		 p = Scalar(123,50,10);
		 break;
	 case 13:
		 p = Scalar(10,240,126);
		 break;
	 case 14:
		 p = Scalar(0,128,255);
		 break;
	 case 15:
		 p = Scalar(128,200,20);
		 break;
	 default:
		 break;
	}

	return p;
}

void EventDetDLL::ClusterCurrentPointsbyDist()
{
	////get cluster from current position
	//for (int i = 0; i < currPointList.size(); i++)
	//{
	//	RepPoint* pt = currPointList[i];
	//	if (groupList[pt->objID] > 0)
	//		continue;
	//	RepTrace* trace = objects.GetTrackbyID(pt->objID);
	//	if (trace == nullptr) continue;
	//	if (trace->outdated)
	//		continue;
	//	groupList[pt->objID] = groupIdx;

	//	bool grouped = false;

	//	for (int j = 0; j < currPointList.size(); j++)
	//	{
	//		if (j == i) continue;
	//		RepPoint* pt1 = currPointList[j];
	//		if (groupList[pt1->objID] > 0) continue;
	//		RepTrace* trace1 = objects.GetTrackbyID(pt1->objID);
	//		if (trace1 == nullptr) continue;
	//		if (trace1->outdated)
	//			continue;
	//		if (RepTrace::timeDiff(pt->ts, pt1->ts) > 5) continue;

	//		float dist = RepTrace::GetDistance(pt, pt1);
	//		if (dist < GroupingDistThre)
	//		{
	//			grouped = true;
	//			groupList[pt1->objID] = groupIdx;
	//		}
	//	}

	//	if (!grouped) 
	//		groupList[pt->objID] = 0;
	//	else
	//		groupIdx ++;
	//}
}

void EventDetDLL::ClusterCurrentPoints()
{
	int pointCount = 0;
	int clusterCount = 5;

	float data[100][2];
	vector<int> IDList;

	for (int i = 0; i < objects.length; i++)
	{
		RepTrace* trace = objects.GetTrack(i);
		if (trace == nullptr) continue;
		if (trace->outdated) continue;

		RepPoint* pt = objects.GetTrack(i)->GetRoot();
		if (pt->objID < 0) continue;
		if (RepTrace::timeDiff(pt->ts, prevTime) <= 0) continue;
		
		data[pointCount][0] = pt->pos.x;
		data[pointCount][1] = pt->pos.y;
		IDList.push_back(pt->objID);
		pointCount ++;
	}

	if (pointCount < clusterCount) return;

	Mat points(pointCount, 2, CV_32FC1, *data);
	Mat labels, centers;

	kmeans(points, clusterCount, labels, TermCriteria (TermCriteria::EPS+TermCriteria::COUNT, 10, 0.0), 3, KMEANS_RANDOM_CENTERS, centers);

	string file = dataFolder+"clusterRes.txt";
	FILE* f = fopen(file.c_str(), "a");
	fprintf(f, "%s\n", currTime.c_str());
	for (int i = 0; i < clusterCount; i++)
	{
		fprintf(f, "Cluster %d:\t", i);
		vector<int> group;
		for (int j = 0; j < IDList.size(); j++)
		{
			if (labels.at<int>(j) == i)
			{
				group.push_back(IDList[j]);
				fprintf(f, "%d ", IDList[j]);
			}
		}
		fprintf(f, "\n");
		if (group.size() > 2)
			CheckGathering(group);
		group.clear();
	}
	fclose(f);
}

typedef float mytype;            // coordinate type
int             d = 2;            // dimension

typedef mytype* const* PointIterator; 
typedef const mytype* CoordIterator;
typedef Miniball::
Miniball <Miniball::CoordAccessor<PointIterator, CoordIterator> > 
MB;

void EventDetDLL::CheckGathering(vector<int> &objs)
{
	//remove short trajectoires
	vector<RepPoint*> root;
	for (int i = 0; i < objs.size(); i++)
	{
		//cout << objs[i];
		RepTrace* track = objects.GetTrackbyID(objs[i]);
		if (track == nullptr)
			continue;

		if (objects.GetTrackbyID(objs[i])->length <= 2)
			continue;

		root.push_back(objects.GetTrackbyID(objs[i])->GetRoot());
	}

	if (root.size() < 2)
		return;

	BBOX box;
	vector<int> ids;
	if (CheckSmallGathering(root, ids, box))
	{
		EventInfo* eve = new EventInfo();
		eve->type = 0;
		eve->endTime = currTime; 
		eve->startTime = RepTrace::timeAddSec(currTime, -60);
		//cout << eve->startTime << " " << currTime << endl;
		eve->objIDs = ids;
		eve->reported = false;
		eve->box = box;
		EventType::SetBit(eve->type, congregationMask);
		events.push_back(eve);
	}

	//vector<RepPoint*> group;
	//group.clear();
	//for (int i = 0; i < root.size(); i++)
	//{
	//	group.push_back(root[i]);
	//	for (int j = 0; j < root.size(); j++)
	//	{
	//		if (j == i) continue;
	//		float dist = RepTrace::GetDistance(root[i], root[j]);
	//		if (dist < GroupingDistThre)
	//			group.push_back(root[j]);
	//	}

	//	if (group.size() > 1)
	//	{
	//		BBOX box;
	//		if (CheckSmallGathering(group, box))
	//		{
	//			EventInfo* eve = new EventInfo();
	//			eve->type = 0;
	//			eve->endTime = currTime; 
	//			eve->startTime = RepTrace::timeAddSec(currTime, -60);
	//			//cout << eve->startTime << " " << currTime << endl;
	//			for (int j = 0; j < group.size(); j++)
	//				eve->objIDs.push_back(group[j]->objID);
	//			eve->reported = false;
	//			eve->box = box;
	//			EventType::SetBit(eve->type, congregationMask);
	//			events.push_back(eve);
	//		}
	//	}
	//}
}

bool EventDetDLL::CheckSmallGathering(vector<RepPoint*> group, vector<int> &ids, BBOX &box)
{

	//if (RepTrace::timeDiff(currTime, "20160628101328") > 0 
	//		&& RepTrace::timeDiff(currTime, "20160628101500") < 0)
	//{
	//	box.bboxL = 86; box.bboxT = 124;
	//	box.bboxW = 180; box.bboxH = 140;
	//	return true;
	//}

	//find the current gathering centre point
	int avgX = 0.0, avgY = 0.0;
	for (int i = 0; i < group.size(); i++)
	{
		avgX += group[i]->pos.x;
		avgY += group[i]->pos.y;
	}
	avgX /= group.size();
	avgY /= group.size();
	RepPoint* centre = new RepPoint(avgX, avgY);

	//find the trends of moving for each object in the group
	int maxTrajLength = 150;
	for (int i = 0; i < group.size(); i++)
	{
		vector<Point> pts;
		pts.clear();
		int cnt = 0;
		RepPoint* prev = group[i];
		int id = prev->objID;
		log4cpp::Category::getInstance(string("sub1")).debug("ID: " + to_string(id));
		while (prev != nullptr)
		{
			float dist = RepTrace::GetDistance(centre, prev);
			log4cpp::Category::getInstance(string("sub1")).debug("  " + to_string(dist));
			if (dist < 30.0) 
			{
				prev = prev->GetPrevPoint();
				cnt++;
				continue;
			}
			pts.push_back(Point((double)cnt, (double)dist));
			if (pts.size() >= maxTrajLength)
				break;
			prev = prev->GetPrevPoint();
			cnt++;
		}

		Line l;
		if (l.fitPoints(pts))
		{
			log4cpp::Category::getInstance(string("sub1")).debug("  slope" + to_string(l._slope));
			if (l._slope > -0.3)
			{
				ids.push_back(id);
			}
		}
		pts.clear();
	}

	if (ids.size() >= minGroupSize) 
	{
		int r = 0, b = 0;
		int l = 9999, t = 9999;

		for (int i = 0; i < group.size(); i++)
		{
			if (std::find(ids.begin(), ids.end(), group[i]->objID) != ids.end())
			{
				if (l > group[i]->box.bboxL) l = group[i]->box.bboxL; //cout << l << " ";
				if (t > group[i]->box.bboxT) t = group[i]->box.bboxT; //cout << t << " ";
				if (r < group[i]->box.bboxL + group[i]->box.bboxW) r = group[i]->box.bboxL + group[i]->box.bboxW; //cout << r << " ";
				if (b < group[i]->box.bboxT + group[i]->box.bboxH) b = group[i]->box.bboxT + group[i]->box.bboxH; //cout << b << endl;
			}
		}

		box.bboxL = l; box.bboxT = t;
		box.bboxW = r-l; box.bboxH = b-t;

		log4cpp::Category::getInstance(string("sub1")).info("Gathering pattern detected at " + to_string(avgX) + "," + to_string(avgY));
	
		return true;
	}
	else
		return false;


	vector<float> radius;
	Scalar color = Scalar(0, 0, 0);

	int n = group.size();
	mytype** ap = new mytype*[n];
	while (true)
	{
		for (int i=0; i<n; i++) 
		{
			mytype* p = new mytype[d];
			p[0] = group[i]->pos.x;
			p[1] = group[i]->pos.y;
			ap[i]=p;
		}
		MB mb (d, ap, ap+n);
		radius.push_back(mb.squared_radius());

		bool invalid = false;
		for (int i = 0; i < n; i++)
		{
			RepPoint* prev = group[i]->GetPrevPoint();
			if (prev == nullptr)
			{
				invalid = true;
				continue;
			}
			group[i] = prev;
		}

		if (invalid) break;
	}

	//clean up
	for (int i=0; i<n; ++i)
		delete[] ap[i];
	delete[] ap;

	if (radius.size() <= 5)
		return false;

	//Jun20 simplified determination
	int count = radius.size();
	if (count > 20) count = 20;
	//int midCount = (int)count/2;
	//if (radius[0] < radius[count-1] && radius[midCount] < radius[count - 1])
	if (radius[count] / radius[0] > 1.5)
	{
		int r = 0, b = 0;
		int l = 9999, t = 9999;

		for (int i = 0; i < group.size(); i++)
		{
			if (l > group[i]->box.bboxL) l = group[i]->box.bboxL; //cout << l << " ";
			if (t > group[i]->box.bboxT) t = group[i]->box.bboxT; //cout << t << " ";
			if (r < group[i]->box.bboxL + group[i]->box.bboxW) r = group[i]->box.bboxL + group[i]->box.bboxW; //cout << r << " ";
			if (b < group[i]->box.bboxT + group[i]->box.bboxH) b = group[i]->box.bboxT + group[i]->box.bboxH; //cout << b << endl;
		}

		box.bboxL = l; box.bboxT = t;
		box.bboxW = r-l; box.bboxH = b-t;

		return true;
	}
	else 
		return false;
}

void EventDetDLL::CalculateDensity()
{
	if (objects.length == 0) 
		return;

	for (int i = 0; i < DENSITYGRIDNUM; i++)
	{
		for (int j = 0; j < DENSITYGRIDNUM; j++)
		{
			currDensity[i][j].count = 0;
			currDensity[i][j].IDList.clear();
			currDensity[i][j].centre.x = 0.0;
			currDensity[i][j].centre.y = 0.0;
		}
	}

	for (int i = 0; i < objects.length; i++)
	{
		RepTrace* trace = objects.GetTrack(i);
		if (trace == nullptr) continue;
		if (trace->outdated) continue;

		RepPoint* pt = objects.GetTrack(i)->GetRoot();
		//cout << pt->objID << endl;
		if (pt->objID < 0) continue;
		if (RepTrace::timeDiff(pt->ts, prevTime) <= 0) continue;
		
		int c = pt->pos.x / minGridSize;
		int r = pt->pos.y / minGridSize;
		currDensity[c+1][r+1].count ++;
		currDensity[c+1][r+1].IDList.push_back(pt->objID);
		currDensity[c+1][r+1].centre.x += pt->pos.x;
		currDensity[c+1][r+1].centre.y += pt->pos.y;

	}

	for (int i = 0; i < DENSITYGRIDNUM; i++)
	{
		for (int j = 0; j < DENSITYGRIDNUM; j++)
		{
			currDensity[i][j].centre.x /= currDensity[i][j].count;
			currDensity[i][j].centre.y /= currDensity[i][j].count;
			if (currDensity[i][j].count > maxDensity[i][j].count)
			{
				maxDensity[i][j].count = currDensity[i][j].count;
				maxDensity[i][j].IDList = currDensity[i][j].IDList;
				maxDensity[i][j].centre.x = currDensity[i][j].centre.x;
				maxDensity[i][j].centre.y = currDensity[i][j].centre.y;
				maxDensityR = j;
				maxDensityC = i;
			}
		}
	}

	string filename = dataFolder+"density.txt";
	FILE* fp = fopen(filename.c_str(), "w");
	fprintf(fp, "\n%s:\n", currTime.c_str());
	for (int i = 0; i < DENSITYGRIDNUM; i++)
	{
		for (int j = 0; j < DENSITYGRIDNUM; j++)
		{
			if (densMap[i][j].size() > 1)
			{
				//if (densMap[i][j].back().count - currDensity[i][j].count > dispersionCountThreshold)
				if (maxDensity[i][j].count > minGroupSize && maxDensity[i][j].count > currDensity[i][j].count)
					CheckDispersion(maxDensity[i][j]);
			}

			fprintf(fp, "%d\t", maxDensity[i][j].count);

			densMap[i][j].push_back(currDensity[i][j]);

			if (densMap[i][j].size() > 15)
			{
				densMap[i][j][0].IDList.clear();
				densMap[i][j].erase(densMap[i][j].begin());

				maxDensity[i][j].count = 0;
				for (int k = 0; k < densMap[i][j].size(); k++)
				{
					if (densMap[i][j][k].count > maxDensity[i][j].count)
					{
						maxDensity[i][j].count = densMap[i][j][k].count;
						maxDensity[i][j].IDList = densMap[i][j][k].IDList;
						maxDensity[i][j].centre.x = densMap[i][j][k].centre.x;
						maxDensity[i][j].centre.y = densMap[i][j][k].centre.y;
					}
				}
			}
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}

void EventDetDLL::ClusterCurrentPointsbyDensity()
{

	if (objects.length <= minGroupSize)
		 return;

	int maxC = maxDensityC, maxR = maxDensityR;

	//from the most dense portion and its neighbors, find the group according to proximity
	vector<int> groupCandidates = currDensity[maxC][maxR].IDList;
	groupCandidates.insert(std::end(groupCandidates), std::begin(currDensity[maxC-1][maxR].IDList), std::end(currDensity[maxC-1][maxR].IDList));
	groupCandidates.insert(std::end(groupCandidates), std::begin(currDensity[maxC+1][maxR].IDList), std::end(currDensity[maxC+1][maxR].IDList));

	groupCandidates.insert(std::end(groupCandidates), std::begin(currDensity[maxC][maxR+1].IDList), std::end(currDensity[maxC][maxR+1].IDList));
	groupCandidates.insert(std::end(groupCandidates), std::begin(currDensity[maxC-1][maxR+1].IDList), std::end(currDensity[maxC-1][maxR+1].IDList));
	groupCandidates.insert(std::end(groupCandidates), std::begin(currDensity[maxC+1][maxR+1].IDList), std::end(currDensity[maxC+1][maxR+1].IDList));

	groupCandidates.insert(std::end(groupCandidates), std::begin(currDensity[maxC-1][maxR-1].IDList), std::end(currDensity[maxC-1][maxR-1].IDList));
	groupCandidates.insert(std::end(groupCandidates), std::begin(currDensity[maxC][maxR-1].IDList), std::end(currDensity[maxC][maxR-1].IDList));
	groupCandidates.insert(std::end(groupCandidates), std::begin(currDensity[maxC+1][maxR+1].IDList), std::end(currDensity[maxC+1][maxR+1].IDList));

	for (int i = 0; i < groupCandidates.size(); i++)
	{
		vector<RepPoint*> group; 
		group.clear();
		RepPoint *pt = objects.GetTrackbyID(groupCandidates[i])->GetRoot();
		group.push_back(pt);
		for (int j = 0; j < groupCandidates.size(); j++)
		{
			if (j == i) continue;
			RepPoint *pt1 = objects.GetTrackbyID(groupCandidates[j])->GetRoot();
			for (int k = 0; k < group.size(); k++)
			{
				if (RepTrace::GetDistance(pt1, group[k]) < groupingDistThreshold)
				{
					group.push_back(pt1);
					break;
				}
			}
		}

		if (group.size() >= minGroupSize)
		{
			BBOX box;
			vector<int> ids;
			if (CheckSmallGathering(group, ids, box))
			{
				//check if this event has been reported (same location)
				if (lastCongregationEvent->box.bboxW > 0)
				{
					if (RepTrace::timeDiff(currTime, lastCongregationEvent->endTime) < congregationIgnoreTime ||
							OverlappedRatio(lastCongregationEvent->box, box) > 0.2)
						continue;
				}
				if (lastCandidate->box.bboxW > 0)
				{
					if (RepTrace::timeDiff(currTime, lastCandidate->endTime) < congregationIgnoreTime ||
							OverlappedRatio(lastCandidate->box, box) > 0.2)
						continue;
				}

				EventInfo* eve = new EventInfo();
				eve->type = 0;
				eve->endTime = currTime; 
				int offset = 0 - updateInterval;
				eve->startTime = RepTrace::timeAddSec(currTime, offset);
				//cout << eve->startTime << " " << currTime << endl;
				eve->objIDs = ids;
				eve->reported = false;
				eve->isCandidate = true;
				eve->box = box;
				EventType::SetBit(eve->type, congregationMask);
				events.push_back(eve);
				lastCandidate->box.bboxL = eve->box.bboxL;
				lastCandidate->box.bboxT = eve->box.bboxT;
				lastCandidate->box.bboxW = eve->box.bboxW;
				lastCandidate->box.bboxH = eve->box.bboxH;
				lastCandidate->endTime = eve->endTime;

				log4cpp::Category::getInstance(string("sub1")).info("Congregation candidate event detected");

			}
			break;
		}
	}

	groupCandidates.clear();
}

void EventDetDLL::CheckDispersion(DensityData densHist)
{
	vector<int> objs = densHist.IDList;
	vector<RepPoint*> group;
	for (int i = 0; i < objs.size(); i++)
	{
		//cout << objs[i];
		RepTrace* track = objects.GetTrackbyID(objs[i]);
		if (track == nullptr)
			continue;

		group.push_back(objects.GetTrackbyID(objs[i])->GetRoot());
	}

	RepPoint* centre = new RepPoint(densHist.centre.x, densHist.centre.y);
	log4cpp::Category::getInstance(string("sub1")).info("Dispersion Centre " + to_string(densHist.centre.x) + "," + to_string(densHist.centre.y));

	//find the trends of moving for each object in the group
	vector<int> ids;
	int maxTrajLength = 30;
	for (int i = 0; i < group.size(); i++)
	{
		vector<Point> pts;
		pts.clear();
		int cnt = 0;
		RepPoint* prev = group[i];
		int id = prev->objID;
		log4cpp::Category::getInstance(string("sub1")).debug("ID: " + to_string(id));
		while (prev != nullptr)
		{
			float dist = RepTrace::GetDistance(centre, prev);
			log4cpp::Category::getInstance(string("sub1")).debug("  " + to_string(dist));
			if (dist < 30.0)
			{
				prev = prev->GetPrevPoint();
				cnt++;
				continue;
			}
			pts.push_back(Point((double)cnt, (double)dist));
			if (pts.size() >= maxTrajLength)
				break;
			prev = prev->GetPrevPoint();
			cnt++;
		}

		Line l;
		if (l.fitPoints(pts))
		{
			log4cpp::Category::getInstance(string("sub1")).debug("  slope:" + to_string(l._slope));
			if (l._slope < -0.5)
			{
				ids.push_back(id);
			}
		}
		pts.clear();
	}

	bool isDispersion = false;
	BBOX box;
	if (ids.size() >= minGroupSize) 
	{
		int r = 0, b = 0;
		int l = 9999, t = 9999;

		for (int i = 0; i < group.size(); i++)
		{
			if (std::find(ids.begin(), ids.end(), group[i]->objID) != ids.end())
			{
				if (l > group[i]->box.bboxL) l = group[i]->box.bboxL; //cout << l << " ";
				if (t > group[i]->box.bboxT) t = group[i]->box.bboxT; //cout << t << " ";
				if (r < group[i]->box.bboxL + group[i]->box.bboxW) r = group[i]->box.bboxL + group[i]->box.bboxW; //cout << r << " ";
				if (b < group[i]->box.bboxT + group[i]->box.bboxH) b = group[i]->box.bboxT + group[i]->box.bboxH; //cout << b << endl;
			}
		}

		box.bboxL = l; box.bboxT = t;
		box.bboxW = r-l; box.bboxH = b-t;
		
		isDispersion = true;
	}

	if (isDispersion)
	{
		//check if this event has been reported (same location)
		if (lastDispersionEvent->box.bboxW > 0)
		{
			//if (RepTrace::timeDiff(currTime, lastDispersionEvent->endTime) < 60) // check
			//	return;
			if (OverlappedRatio(lastDispersionEvent->box, box) > 0.2)
				return;
		}

		EventInfo* eve = new EventInfo();
		eve->type = 0;
		eve->endTime = currTime; 
		int offset = 0 - updateInterval;
		eve->startTime = RepTrace::timeAddSec(currTime, offset);
		eve->objIDs = ids;
		eve->reported = false;
		eve->box = box;
		EventType::SetBit(eve->type, dispersionMask);
		events.push_back(eve);

		lastDispersionEvent->box.bboxL = eve->box.bboxL;
		lastDispersionEvent->box.bboxT = eve->box.bboxT;
		lastDispersionEvent->box.bboxW = eve->box.bboxW;
		lastDispersionEvent->box.bboxH = eve->box.bboxH;
		lastDispersionEvent->endTime = eve->endTime;

		log4cpp::Category::getInstance(string("sub1")).info("Dispersion event detected");
	}

	return;

	//vector<float> radius;
	//int n = group.size();
	//mytype** ap = new mytype*[n];
	//while (true)
	//{
	//	for (int i=0; i<n; i++) 
	//	{
	//		mytype* p = new mytype[d];
	//		p[0] = group[i]->pos.x;
	//		p[1] = group[i]->pos.y;
	//		ap[i]=p;
	//	}
	//	MB mb (d, ap, ap+n);
	//	radius.push_back(mb.squared_radius());

	//	bool invalid = false;
	//	for (int i = 0; i < n; i++)
	//	{
	//		RepPoint* prev = group[i]->GetPrevPoint();
	//		if (prev == nullptr)
	//		{
	//			invalid = true;
	//			continue;
	//		}
	//		group[i] = prev;
	//	}

	//	if (invalid) break;
	//}

	////clean up
	//for (int i=0; i<n; ++i)
	//	delete[] ap[i];
	//delete[] ap;

	//if (radius.size() <= 5)
	//	return;

	//bool isDispersion = false;
	//BBOX box;
	////Jun20 simplified determination
	//int count = radius.size();
	//if (count > 15) count = 15;
	//int midCount = (int)count/2;
	////if (radius[0] > radius[count-1]) && radius[midCount] > radius[count - 1])
	//if (radius[0] / radius[5] > 1.15)
	//{
	//	int r = 0, b = 0;
	//	int l = 9999, t = 9999;

	//	for (int i = 0; i < group.size(); i++)
	//	{
	//		if (l > group[i]->box.bboxL) l = group[i]->box.bboxL; //cout << l << " ";
	//		if (t > group[i]->box.bboxT) t = group[i]->box.bboxT; //cout << t << " ";
	//		if (r < group[i]->box.bboxL + group[i]->box.bboxW) r = group[i]->box.bboxL + group[i]->box.bboxW; //cout << r << " ";
	//		if (b < group[i]->box.bboxT + group[i]->box.bboxH) b = group[i]->box.bboxT + group[i]->box.bboxH; //cout << b << endl;
	//	}

	//	box.bboxL = l; box.bboxT = t;
	//	box.bboxW = r-l; box.bboxH = b-t;

	//	isDispersion =  true;
	//}

	//if (isDispersion)
	//{
	//	//check if this event has been reported (same location)
	//	if (lastDispersionEvent->box.bboxW > 0)
	//	{
	//		//if (RepTrace::timeDiff(currTime, lastDispersionEvent->endTime) < 60) // check
	//		//	return;
	//		if (OverlappedRatio(lastDispersionEvent->box, box) > 0.2)
	//			return;
	//	}

	//	EventInfo* eve = new EventInfo();
	//	eve->type = 0;
	//	eve->endTime = currTime; 
	//	int offset = 0 - updateInterval;
	//	eve->startTime = RepTrace::timeAddSec(currTime, offset);
	//	for (int j = 0; j < group.size(); j++)
	//		eve->objIDs.push_back(group[j]->objID);
	//	eve->reported = false;
	//	eve->box = box;
	//	EventType::SetBit(eve->type, dispersionMask);
	//	events.push_back(eve);

	//	lastDispersionEvent->box.bboxL = eve->box.bboxL;
	//	lastDispersionEvent->box.bboxT = eve->box.bboxT;
	//	lastDispersionEvent->box.bboxW = eve->box.bboxW;
	//	lastDispersionEvent->box.bboxH = eve->box.bboxH;
	//	lastDispersionEvent->endTime = eve->endTime;

	//}
}

void EventDetDLL::CheckCandidateEvents()
{
	for (int i = 0; i < events.size(); i++)
	{
		EventInfo* eve = events[i];
		if (!eve->isCandidate)
			continue;
		if (EventType::GetBit(eve->type, EventDetection::congregationMask) == 1)
		{
			if (RepTrace::timeDiff(currTime, eve->endTime) < congregationStayTime)
			{
				continue;
			}

			vector<RepPoint*> group; 
			group.clear();
			for (int j = 0; j < eve->objIDs.size(); j++)
			{
				RepPoint *pt = objects.GetTrackbyID(eve->objIDs[j])->GetRoot();
				group.push_back(pt);
			}

			BBOX box;
			vector<int> ids;
			if (CheckSmallGathering(group, ids, box) && OverlappedRatio(box, eve->box) > 0.5)
			{
				if (lastCongregationEvent->box.bboxW > 0)
				{
					if (RepTrace::timeDiff(eve->endTime, lastCongregationEvent->endTime) < congregationIgnoreTime || 
						OverlappedRatio(lastCongregationEvent->box, box) > 0.2)
						continue;
				}

				eve->box = box;
				eve->objIDs.clear();
				eve->objIDs = ids;
				eve->endTime = currTime;
				eve->isCandidate = false;

				lastCongregationEvent->box.bboxL = eve->box.bboxL;
				lastCongregationEvent->box.bboxT = eve->box.bboxT;
				lastCongregationEvent->box.bboxW = eve->box.bboxW;
				lastCongregationEvent->box.bboxH = eve->box.bboxH;
				lastCongregationEvent->endTime = eve->endTime;
				log4cpp::Category::getInstance(string("sub1")).info("Congregation candidate confirmed");
			}
			else
			{
				eve->reported = true;
				lastCandidate->box.bboxW = 0;
				log4cpp::Category::getInstance(string("sub1")).info("Congregation candidate dropped");
			}
		}
	}
}

void EventDetDLL::UpdateStatus()
{
	log4cpp::Category::getInstance(string("sub1")).info("Timestamp: " + currTime);

	hasEvents = false;
	metaFrame = Scalar(255, 255, 255);

	//remove reported events;
	for (int i = 0; i < events.size(); i++)
	{
		if (events[i]->reported) 
			events.erase(events.begin() + i);
	}

	//remove outdated tracks
	ClearOutdatedTracks();

	vector<pair<int, float>> SpeedList;
	float maxSpeed = 0.0;

	for (int i = 0; i < objects.length; i++)
	{
		RepTrace* trace = objects.GetTrack(i);

		if (trace == nullptr)
			continue;
		
		if (trace->outdated)
			continue;

		trace->UpdateStatus();

		if (EventType::GetBit(trace->eventStatus, loiterMask) == 1 && !trace->loiterReported && EventType::GetBit(reportEventMask, loiterMask) == 1)
		{
			char hour[2];
			hour[0] = currTime.c_str()[8];
			hour[1] = currTime.c_str()[9];
			if (atoi(hour) >= loiterFrom || atoi(hour) < loiterTill)
			{
				EventInfo* eve = new EventInfo();
				eve->isCandidate = false;
				eve->reported = false;
				eve->type = 0;
				EventType::SetBit(eve->type, loiterMask);
				eve->endTime = trace->GetRoot()->ts;
				eve->startTime = RepTrace::timeAddSec(trace->GetLastRP()->ts, -60);
				eve->objIDs.push_back(trace->ID);
				eve->box = trace->GetRoot()->box;
				events.push_back(eve);
			}
			trace->loiterReported = true;
		}

		//abnormal loitering
		if (trace->directionChangeCount > 5)
		{
			float loiterTime = RepTrace::GetDuration(trace->GetLastRP(), trace->GetFirstRP());
			if (loiterTime / trace->duration > 0.5)
			{
				EventInfo* eve = new EventInfo();
				eve->isCandidate = false;
				eve->reported = false;
				eve->type = 0;
				EventType::SetBit(eve->type, abLoiterMask);
				eve->endTime = trace->GetLastRP()->ts;
				eve->startTime = RepTrace::timeAddSec(trace->GetLastRP()->ts, -60);
				eve->box = trace->GetRoot()->box;
				eve->objIDs.push_back(trace->ID);
				events.push_back(eve);
			}	
		}

		//abnormal running
		if (trace->curSpeed > RepTrace::runningSpeedThreshold && trace->maxAcceleration > RepTrace::runningAccelerationThreshold)
		{
			EventInfo* eve = new EventInfo();
			eve->isCandidate = false;
			eve->reported = false;
			eve->type = 0;
			EventType::SetBit(eve->type, runningMask);
			eve->endTime = trace->GetRoot()->ts;
			eve->startTime = RepTrace::timeAddSec(trace->GetLastRP()->ts, -60);
			eve->objIDs.push_back(trace->ID);
			eve->box = trace->GetRoot()->box;
			events.push_back(eve);
		}

		//grouping
		//SpeedList.push_back(make_pair(trace->GetRoot()->objID, trace->curSpeed));
		//if (trace->curSpeed > maxSpeed)
		//	maxSpeed = trace->curSpeed;
		//
		//if (EventType::GetBit(trace->eventStatus, intrusionMask) == 1)
		//{
		//	EventInfo* eve = new EventInfo();
		//	eve->reported = false;
		//	eve->type = 0;
		//	EventType::SetBit(eve->type, intrusionMask);
		//	eve->startTime = trace->GetFirstPoint()->ts;
		//	eve->endTime = trace->GetRoot()->ts;
		//	eve->objIDs.push_back(trace->ID);
		//	events.push_back(eve);
		//}

		//check if the trace is outdated -> object moved out already
		trace->CheckOutdateStatus(this->currTime);
	}

	CheckCandidateEvents();
	//cout << "Checked candidate events" << endl;

	//ClusterCurrentPoints();
	CalculateDensity();
	ClusterCurrentPointsbyDensity();

	if (objects.length > 0)
	{
		for (int i = 0; i < objects.length; i++)
		{
			RepTrace* trace = objects.GetTrack(i);
			if (trace->outdated || trace->length == 1)
				continue;
			RepPoint* point = trace->GetRoot();
			RepPoint* prevPoint = point;
			Scalar color = colorList[0];
			int count = trace->length;
			while (count > 0)
			{
				circle(metaFrame, cv::Point((int)point->pos.x, (int)point->pos.y), 3, color, -1);
				prevPoint = point;
				point = point->GetPrevPoint();
				if (count >= 2)
					line(metaFrame, cvPoint((int)point->pos.x, (int)point->pos.y), cvPoint((int)prevPoint->pos.x, (int)prevPoint->pos.y), color);
				count --;
				if (trace->length - count > 10) break;
			}

			circle(metaFrame, cv::Point((int)trace->GetRoot()->pos.x, (int)trace->GetRoot()->pos.y), 3, Scalar(0, 0, 0), -1);
			putText(metaFrame, to_string(trace->GetRoot()->objID), cv::Point((int)trace->GetRoot()->pos.x, (int)trace->GetRoot()->pos.y), CV_FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 0), 1);
		}

	}

	char info[200];
	for (int i = 0; i < events.size(); i++)
	{
		EventInfo* eve = events[i];
		if (EventType::GetBit(eve->type, loiterMask) == 1)
			sprintf(info, "Loiter %s %d", eve->startTime.c_str(), eve->objIDs[0]);
		else if (EventType::GetBit(eve->type, runningMask) == 1)
			sprintf(info, "Running %s %d", eve->startTime.c_str(), eve->objIDs[0]);
		else if (EventType::GetBit(eve->type, congregationMask) == 1)
		{
			sprintf(info, "Congregation %s ", eve->startTime.c_str());
			for (int j = 0; j < eve->objIDs.size(); j++)
			{
				sprintf(info, "%s%d ", info, eve->objIDs[j]);
				RepTrace* trace = objects.GetTrackbyID(eve->objIDs[j]);
				if (trace == nullptr) continue;
				RepPoint* point = trace->GetRoot();
				RepPoint* prevPoint = point;
				Scalar color = colorList[5];
				int count = trace->length;
				while (count > 0)
				{
					circle(metaFrame, cv::Point((int)point->pos.x, (int)point->pos.y), 3, color, -1);
					prevPoint = point;
					point = point->GetPrevPoint();
					if (count >= 2)
						line(metaFrame, cvPoint((int)point->pos.x, (int)point->pos.y), cvPoint((int)prevPoint->pos.x, (int)prevPoint->pos.y), color);
					count --;
					if (trace->length - count > 10) break;
				}
			}
		}
		else if (EventType::GetBit(eve->type, dispersionMask) == 1)
		{
			sprintf(info, "Dispersion %s ", eve->startTime.c_str());
			for (int j = 0; j < eve->objIDs.size(); j++)
			{
				sprintf(info, "%s%d ", info, eve->objIDs[j]);
				RepTrace* trace = objects.GetTrackbyID(eve->objIDs[j]);
				if (trace == nullptr) continue;
				RepPoint* point = trace->GetRoot();
				RepPoint* prevPoint = point;
				Scalar color = colorList[3];
				int count = trace->length;
				while (count > 0)
				{
					circle(metaFrame, cv::Point((int)point->pos.x, (int)point->pos.y), 3, color, -1);
					prevPoint = point;
					point = point->GetPrevPoint();
					if (count >= 2)
						line(metaFrame, cvPoint((int)point->pos.x, (int)point->pos.y), cvPoint((int)prevPoint->pos.x, (int)prevPoint->pos.y), color);
					count --;
					if (trace->length - count > 10) break;
				}
			}
		}

		//log4cpp::Category::getInstance(string("sub1")).info(string(info));
	}

#ifdef DEBUG
	fclose(trackFile);
#endif

	for (int i = 0; i < events.size(); i++)
	{
		EventInfo* eve = events[i];
		if (!eve->isCandidate && !eve->reported && EventType::GetBit(eve->type, reportEventMask) > 0)
		{
			hasEvents = true;
			break;
		}
	}

	processBatchID ++;

	return;

	//get the histogram of speed
	const double bucket_size = 0.5;
	double area=0;
	int number_of_buckets = (int)ceil(maxSpeed / bucket_size); 
	vector<vector<pair<int, float>>> histogram(number_of_buckets);
	vector<vector<int>> groups; 
	int idxGroup = 0;
	for (int i = 0; i < SpeedList.size(); i++)
	{
		int bucket = (int)floor(SpeedList[i].second / bucket_size);
		histogram[bucket].push_back(SpeedList[i]);
	}

	for (int i = 0; i < histogram.size(); i++)
	{
		bool formGroup = false;
		if (histogram[i].size() >= 2)
		{
			vector<int> toRemove;
			for (int j = 0; j < histogram[i].size(); j++)
			{
				RepTrace* trace = objects.GetTrackbyID(histogram[i][j].first);
				float distToOthers = 0.0;
				for (int k = j+1; k< histogram[i].size(); k++)
				{
					RepTrace* trace1 = objects.GetTrackbyID(histogram[i][k].first);
					distToOthers += RepTrace::GetDuration(trace->GetRoot(), trace1->GetRoot());
				}
				distToOthers /= histogram[i].size() -1;
				if (distToOthers > groupingDistThreshold)
					toRemove.push_back(j);
			}

			if (histogram[i].size() - toRemove.size() >= 2)
			{
				formGroup = true;
				for (int j = 0; j < histogram[i].size(); j++)
				{
					if (find(toRemove.begin(), toRemove.end(), j) != toRemove.end())
						continue;			
					groups[idxGroup].push_back(j);
				}
			}			
		}
		if (formGroup) idxGroup ++;
	}

}

DWORD WINAPI thUpdateStatus(LPVOID lpParam) 
{
	while (true)
	{
		pthis->isUpdating = true;
		if (pthis->bufferSwitch == 0)
		{
			pthis->bufferSwitch = 1;
			pthis->UpdatePoints(pthis->PointBuffer);
		}
		else if (pthis->bufferSwitch == 1)
		{
			pthis->bufferSwitch = 0;
			pthis->UpdatePoints(pthis->PointBuffer_1);
		}

		pthis->UpdateStatus();

		pthis->isUpdating = false;

		if (pthis->bufferSwitch == 1)
			pthis->PointBuffer.clear();
		else if (pthis->bufferSwitch == 0)
			pthis->PointBuffer_1.clear();

		_sleep(pthis->updateInterval);
	}
}

void EventDetDLL::ReportEvents(vector<EventInfo*> &eventList)
{
	for (int i = 0; i < events.size(); i++)
	{
		EventInfo* eve = events[i];
		if (!eve->reported && EventType::GetBit(eve->type, reportEventMask) > 0)
		{
			eventList.push_back(eve);
			eve->reported = true;
		}
	}
}

void EventDetDLL::UpdatePoints(vector<RepPoint*> &buffer)
{
	for (int i = 0; i < buffer.size(); i++)
	{
		RepPoint *obj = buffer[i];
		RepTrace *track = objects.GetTrackbyID(obj->objID);
		if (track == nullptr)
		{
			track = new RepTrace();
			track->AppendPoint(obj);
			track->ID = obj->objID;
			objects.AddTrack(track);
		}
		else
		{
			track->AppendPoint(obj);
		}
	}
}

void EventDetDLL::UpdateFrame(Mat img, string time)
{
	if (!isUpdating)
	{
		currFrame = img;
		prevTime = currTime;
		currTime = time;
		//cout << "EventDet Current Timestamp:" << currTime << endl;
	}
}

void toLowerCase(string &input)
{
	transform(input.begin(), input.end(), input.begin(), ::tolower);
}

void EventDetDLL::LoadConfig()
{
	reportEventMask = 0;
	saveOutput = false;

	string filename = dataFolder+"EventDetConfig.txt";
	ifstream file(filename);
	string oneline;
	getline(file, oneline);
	while (!file.eof())
	{
		istringstream iss(oneline);
		string paraName, value;
		iss >> paraName;
		iss >> value;
		toLowerCase(paraName);
		if (paraName == "updateinterval")
			updateInterval = atoi(value.c_str());
		if (paraName == "mingridsize")
			minGridSize = atoi(value.c_str());
		if (paraName == "mingroupsize")
			minGroupSize = atoi(value.c_str());
		if (paraName == "eventmask")
		{
			string filename = dataFolder+value;
			eventMask = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
			resize(eventMask, eventMask, Size(width, height));
		}
		if (paraName == "reportloiteringfrom")
			loiterFrom = atoi(value.c_str());
		if (paraName == "reportloiteringtill")
		{
			loiterTill = atoi(value.c_str());
			if (loiterTill < loiterFrom) loiterTill += 24;
		}
		if (paraName == "congregationstaytime")
			congregationStayTime = atoi(value.c_str());
		if (paraName == "congregationignoretime")
			congregationIgnoreTime = atoi(value.c_str());
		if (paraName == "groupingdist")
			groupingDistThreshold = atof(value.c_str());
		if (paraName == "dispersiondropcount")
			dispersionCountThreshold = atoi(value.c_str());
		if (paraName == "outdatedtime")
			RepTrace::outdatedTimeThreshold = atoi(value.c_str());
		if (paraName == "loiterstaytime")
			RepTrace::loiterStayTimeThreshold = atoi(value.c_str());
		if (paraName == "runningspeed")
			RepTrace::runningSpeedThreshold = atof(value.c_str());
		if (paraName == "runningacceleration")
			RepTrace::runningAccelerationThreshold = atof(value.c_str());
		if (paraName == "reportloitering" && value == "1")
			EventType::SetBit(reportEventMask, loiterMask);
		if (paraName == "reportcongregation" && value == "1")
			EventType::SetBit(reportEventMask, congregationMask);
		if (paraName == "reportdispersion" && value == "1")
			EventType::SetBit(reportEventMask, dispersionMask);
		if (paraName == "reportabloitering" && value == "1")
			EventType::SetBit(reportEventMask, abLoiterMask);
		if (paraName == "reportrunning" && value == "1")
			EventType::SetBit(reportEventMask, runningMask);
		if (paraName == "saveoutput" && value == "1")
			saveOutput = true;
		getline(file, oneline);
	}
	file.close();
}

void EventDetDLL::StopReporting(int mask)
{
	EventType::ClearBit(reportEventMask, mask);
}

void EventDetDLL::LoadIntrusionSetting()
{
	//intrusionMapMask= imread(".\\intrusionMask.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	//ifstream in("intrusion.cfg");
	//if (!in)
	//{
	//	cout<<"Could not load aggression config file."<<endl;
	//	return;
	//}
	//char junk[1000];
	//while (!in.eof)
	//{
	//	Line line;
	//	in.getline(junk, 1000);
	//	in >> junk;
	//	in >> line.start.x;
	//	in >> line.start.y;
	//	in >> line.end.x;
	//	in >> line.end.y;
	//	intrusionList.push_back(line);
	//}
}

void EventDetDLL::StartThread()
{
	CreateThread(NULL, 0, thUpdateStatus, NULL, 0, NULL);
}

EventDetDLL::EventDetDLL(){;}

EventDetDLL::EventDetDLL(int procW, int procH, string folder)
{

	width = procW; height = procH;
	metaFrame.create(height, width, CV_8UC3);
	dataFolder = folder;

	string logFile = dataFolder + "log.property";
	cout << logFile << endl;
	log4cpp::PropertyConfigurator::configure(logFile);
	//log4cpp::PropertyConfigurator::configure("log.property");

	colorList.push_back(Scalar(0, 0, 255));
	colorList.push_back(Scalar(0, 128, 255));
	colorList.push_back(Scalar(0, 255, 255));
	colorList.push_back(Scalar(0, 255, 0));
	colorList.push_back(Scalar(255, 255, 0));
	colorList.push_back(Scalar(255, 0, 0));
	colorList.push_back(Scalar(255, 0, 128));
	colorList.push_back(Scalar(0, 153, 153));
	colorList.push_back(Scalar(0, 0, 0));

	LoadConfig();
	lastCongregationEvent = new EventInfo();
	lastCongregationEvent->box.bboxW = 0;
	lastCandidate = new EventInfo();
	lastCandidate->box.bboxW = 0;
	lastDispersionEvent = new EventInfo();
	lastDispersionEvent->box.bboxW = 0;

	candidateEvents.clear();

	for (int i = 0; i < DENSITYGRIDNUM; i++)
	{
		for (int j = 0; j < DENSITYGRIDNUM; j++)
		{
			maxDensity[i][j].count = 0;
		}
	}

	//LoadIntrusionSetting();
	objects.length = 0;
	lastEventID = 0;
	pthis = this;
	bufferSwitch = 0;
	for (int i = 0; i < PointBuffer.size(); i++)
		delete PointBuffer[i];
	PointBuffer.clear();
	for (int i = 0; i < PointBuffer_1.size(); i++)
		delete PointBuffer_1[i];
	PointBuffer_1.clear();
	processBatchID = 0;

	StartThread();
}

EventDetDLL::~EventDetDLL()
{
	for (int i = 0; i < objects.length; i++)
		objects.DeleteTrack(i);
	for (int i = 0; i < events.size(); i++)
		delete events[i];
	events.clear();
}

bool EventDetDLL::CheckIntrusion(double x, double y)
{
	int col[2]; col[0] = int(x); col[1] = ceil(x);
	int row[2]; row[0] = int(y); row[2] = ceil(y);
	bool intruded = true;
	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 2; j++)
		{
			Scalar pixel = intrusionMapMask.at<uchar>(row[i], col[j]);
			intruded &= (pixel[0] == 0);
		}
	return intruded;
}

void EventDetDLL::AddPoints(BBOX box, int id, string ts, int frameID, int camID)
{
	double value = sum(eventMask(Range(box.bboxT, box.bboxT+box.bboxH-1), Range(box.bboxL, box.bboxL+box.bboxW-1)))[0];
	if (value / (box.bboxW * box.bboxH) < 100)
	{
		return;
	}
	RepPoint *obj = new RepPoint(box, ts, frameID, camID, id);

	if (bufferSwitch == 0)
	{
		PointBuffer.push_back(obj);
	}
	else if (bufferSwitch == 1)
	{
		PointBuffer_1.push_back(obj);
	}
}



