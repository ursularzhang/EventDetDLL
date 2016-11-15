// This is the main DLL file.

#include "stdafx.h"
#include "EventDetDll.h"
#include <time.h>
#include <utility>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace EventDetection;

//#define DEBUG
EventDetDLL* pthis;

typedef struct DensityData
{
	int count;
	vector<int> IDList;
};

typedef vector<DensityData> DensityHist;

int						maxDensityR, maxDensityC;
DensityData				currDensity[20][20];
DensityHist				densMap[20][20];


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
#ifdef DEBUG
	FILE *fp = fopen(".\\output\\speedSummary.txt", "a");
#endif
	for (int i = 0; i < objects.length; i++)
	{
		RepTrace *trace = objects.GetTrack(i);
		if (trace->outdated)
		{
#ifdef DEBUG
			fprintf(fp, "%d\t%.3f\t%.3f\n", trace->ID, trace->avgSpeed, trace->maxAcceleration);
#endif
			objects.DeleteTrack(i);
		}
	}
#ifdef DEBUG
	fclose(fp);
#endif
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

//
//#include "Clustering.h"
//extern vector<InputFeature> InputVector;
//extern const int K;
//extern const int dim;
//extern void k_means(double U[K][dim]);
//
//void EventDetDLL::ClusterCurrentPoints()
//{
//	InputVector.clear();
//
//	for (int i = 0; i < objects.length; i++)
//	{
//		RepTrace* trace = objects.GetTrack(i);
//		if (trace == nullptr) continue;
//		if (trace->outdated) continue;
//
//		RepPoint* pt = objects.GetTrack(i)->GetRoot();
//		if (pt->objID < 0) continue;
//		if (RepTrace::timeDiff(pt->ts, prevTime) <= 0) continue;
//		//if (RepTrace::timeDiff(pt->ts, currTime) > 2) continue;
//
//		double xt[2];
//		xt[0] = pt->pos.x; xt[1] = pt->pos.y;
//		InputFeature x(xt, pt->objID);
//		InputVector.push_back(x);
//	}
//
//	double U[K][dim]={{250,200},{500, 200},{750,200},{1000, 200}, {1250, 200},
//					{250,400},{500, 400},{750,400},{1000, 400}, {1250, 400},
//					{250,600},{500, 600},{750,600},{1000, 600}, {1250, 600},
//					{250,800},{500, 800},{750,800},{1000, 800}, {1250, 800}};
//	k_means(U);
//
//	//cout << currTime << endl;
//	//check gathering pattern
//	for (int i = 0; i < K; i++)
//	{
//		string filename = ".\\eventOutput\\cluster_" + to_string(i) + ".jpg";
//		Mat frame = currFrame.clone();
//		vector<int> group;
//		for (int j = 0; j < InputVector.size(); j++)
//		{
//			Scalar color = GetDrawColor(i);
//			if (InputVector[j].cluster == i)
//			{
//				group.push_back(InputVector[j].objID);
//
//				RepTrace* trace = objects.GetTrackbyID(InputVector[j].objID);
//				BBOX box = trace->GetRoot()->box;
//				rectangle(frame, cv::Point(box.bboxL, box.bboxT), cv::Point(box.bboxL + box.bboxW, box.bboxT + box.bboxH), color, -1);
//			}
//			imwrite(filename, frame);
//		}
//
//		CheckGathering(group);
//		group.clear();
//	}
//}

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
	if (CheckSmallGathering(root, box))
	{
		EventInfo* eve = new EventInfo();
		eve->type = 0;
		eve->endTime = currTime; 
		eve->startTime = RepTrace::timeAddSec(currTime, -60);
		//cout << eve->startTime << " " << currTime << endl;
		for (int j = 0; j < root.size(); j++)
			eve->objIDs.push_back(root[j]->objID);
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

bool EventDetDLL::CheckSmallGathering(vector<RepPoint*> group, BBOX &box)
{
	vector<float> radius;
	Scalar color = Scalar(0, 0, 0);

	//FILE* f = fopen(".\\eventOutput\\Gathering.txt", "a");
	//fprintf(f, "%s\n", currTime.c_str());
	int n = group.size();
	mytype** ap = new mytype*[n];
	while (true)
	{
		for (int i=0; i<n; i++) 
		{
			mytype* p = new mytype[d];
			p[0] = group[i]->pos.x;
			p[1] = group[i]->pos.y;
			//fprintf(f, "\t%d\t%s\t%.3f\t%.3f\n", group[i]->objID, group[i]->ts.c_str(), p[0], p[1]);
			//cout << group[i]->objID << " " << group[i]->ts.c_str() << " " << p[0] << " " << p[1] << endl;
			ap[i]=p;
		}
		MB mb (d, ap, ap+n);
		//fprintf(f, "Radius %.3f\n", mb.squared_radius());
		//cout << mb.squared_radius() << endl;
		//if (mb.squared_radius() > GroupingDistThre) 
		//	break;
		radius.push_back(mb.squared_radius());

		bool invalid = false;
		for (int i = 0; i < n; i++)
		{
			//cvCircle(img, cv::Point((int)root[i]->pos.x, (int)root[i]->pos.y), 3, color, -1);
			RepPoint* prev = group[i]->GetPrevPoint();
			//RepPoint* prev = group[i]->GetPrevPointbySec(1);
			if (prev == nullptr)
			{
				invalid = true;
				continue;
			}
			group[i] = prev;
			//cvLine(img, cv::Point((int)prev->pos.x, (int)prev->pos.y), cv::Point((int)root[i]->pos.x, (int)root[i]->pos.y), color);
			//cvCircle(img, cv::Point((int)prev->pos.x, (int)prev->pos.y), 3, color, -1);
		}

		if (invalid) break;
	}

	//clean up
	for (int i=0; i<n; ++i)
		delete[] ap[i];
	delete[] ap;

	if (radius.size() <= 5)
		return false;

	//fclose(f);
	//Jun20 simplified determination
	int count = radius.size();
	int midCount = (int)count/2;
	if (radius[0] < radius[count-1] && radius[midCount] < radius[count - 1])
	{
		int r = 0, b = 0;
		int l = 9999, t = 9999;

		for (int i = 0; i < group.size(); i++)
		{
			//cout << group[i]->box.bboxL << " " << group[i]->box.bboxT << " " << group[i]->box.bboxW << " " << group[i]->box.bboxH << endl;
			if (l > group[i]->box.bboxL) l = group[i]->box.bboxL; //cout << l << " ";
			if (t > group[i]->box.bboxT) t = group[i]->box.bboxT; //cout << t << " ";
			if (r < group[i]->box.bboxL + group[i]->box.bboxW) r = group[i]->box.bboxL + group[i]->box.bboxW; //cout << r << " ";
			if (b < group[i]->box.bboxT + group[i]->box.bboxH) b = group[i]->box.bboxT + group[i]->box.bboxH; //cout << b << endl;
		}

		//cout << l << " " << t << " " << r << " " << b << endl;
		box.bboxL = l; box.bboxT = t;
		box.bboxW = r-l; box.bboxH = b-t;

		return true;
	}
	else 
		return false;

	//int decreaseCount = 0;
	//for (int i = 0; i < radius.size()-1; i++)
	//{
	//	if (radius[i] < radius[i+1])
	//		decreaseCount ++;
	//}
	//fprintf(f, "\t\tdecreaseCount: %d in %d\n", decreaseCount, radius.size());
	//fclose(f);

	//if ((float) decreaseCount / radius.size() > 0.75)
	//{
	//	//cvSaveImage(".\\output\\gathering.jpg", img);
	//	//cvReleaseImage(&img);
	//	return true;
	//}
	//else
	//{
	//	//cvSaveImage(".\\output\\non_gathering.jpg", img);
	//	//cvReleaseImage(&img);
	//	return false;
	//}
}

void EventDetDLL::CalculateDensity()
{

	if (objects.length == 0) 
		return;

	int maxCount = 0;

	for (int i = 0; i < 20; i++)
	{
		for (int j = 0; j < 20; j++)
		{
			currDensity[i][j].count = 0;
			currDensity[i][j].IDList.clear();
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
		if (currDensity[c+1][r+1].count > maxCount)
		{
			maxCount = currDensity[c+1][r+1].count;
			maxDensityR = r+1;
			maxDensityC = c+1;
		}
	}

	for (int i = 0; i < 20; i++)
	{
		for (int j = 0; j < 20; j++)
		{
			if (densMap[i][j].size() > 1)
			{
				if (densMap[i][j].back().count - currDensity[i][j].count > DispersionThre)
					CheckDispersion(densMap[i][j].back().IDList);
			}

			densMap[i][j].push_back(currDensity[i][j]);

			if (densMap[i][j].size() > 20)
			{
				densMap[i][j][0].IDList.clear();
				densMap[i][j].erase(densMap[i][j].begin());
			}
		}
	}
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
				if (RepTrace::GetDistance(pt1, group[k]) < GroupingDistThre)
				{
					group.push_back(pt1);
					break;
				}
			}
		}

		if (group.size() >= minGroupSize)
		{

			BBOX box;
			if (CheckSmallGathering(group, box))
			{
				//check if this event has been reported (same location)
				if (lastCongregationEvent->box.bboxW > 0)
				{
					if (RepTrace::timeDiff(currTime, lastCongregationEvent->endTime) < 60) // check
						continue;
					if (OverlappedRatio(lastCongregationEvent->box, box) > 0.2)
						continue;
				}

				EventInfo* eve = new EventInfo();
				eve->type = 0;
				eve->endTime = currTime; 
				eve->startTime = RepTrace::timeAddSec(currTime, -60);
				//cout << eve->startTime << " " << currTime << endl;
				for (int j = 0; j < group.size(); j++)
					eve->objIDs.push_back(group[j]->objID);
				eve->reported = false;
				eve->box = box;
				EventType::SetBit(eve->type, congregationMask);
				events.push_back(eve);

				lastCongregationEvent->box.bboxL = eve->box.bboxL;
				lastCongregationEvent->box.bboxT = eve->box.bboxT;
				lastCongregationEvent->box.bboxW = eve->box.bboxW;
				lastCongregationEvent->box.bboxH = eve->box.bboxH;
				lastCongregationEvent->endTime = eve->endTime;
			}
			break;
		}
	}

	groupCandidates.clear();
}

void EventDetDLL::CheckDispersion(vector<int> &objs)
{
	vector<RepPoint*> group;
	for (int i = 0; i < objs.size(); i++)
	{
		//cout << objs[i];
		RepTrace* track = objects.GetTrackbyID(objs[i]);
		if (track == nullptr)
			continue;

		group.push_back(objects.GetTrackbyID(objs[i])->GetRoot());
	}

	vector<float> radius;

	//FILE* f = fopen(".\\eventOutput\\Gathering.txt", "a");
	//fprintf(f, "%s\n", currTime.c_str());
	int n = group.size();
	mytype** ap = new mytype*[n];
	while (true)
	{
		for (int i=0; i<n; i++) 
		{
			mytype* p = new mytype[d];
			p[0] = group[i]->pos.x;
			p[1] = group[i]->pos.y;
			//fprintf(f, "\t%d\t%s\t%.3f\t%.3f\n", group[i]->objID, group[i]->ts.c_str(), p[0], p[1]);
			//cout << group[i]->objID << " " << group[i]->ts.c_str() << " " << p[0] << " " << p[1] << endl;
			ap[i]=p;
		}
		MB mb (d, ap, ap+n);
		//fprintf(f, "Radius %.3f\n", mb.squared_radius());
		//cout << mb.squared_radius() << endl;
		//if (mb.squared_radius() > GroupingDistThre) 
		//	break;
		radius.push_back(mb.squared_radius());

		bool invalid = false;
		for (int i = 0; i < n; i++)
		{
			//cvCircle(img, cv::Point((int)root[i]->pos.x, (int)root[i]->pos.y), 3, color, -1);
			RepPoint* prev = group[i]->GetPrevPoint();
			//RepPoint* prev = group[i]->GetPrevPointbySec(1);
			if (prev == nullptr)
			{
				invalid = true;
				continue;
			}
			group[i] = prev;
			//cvLine(img, cv::Point((int)prev->pos.x, (int)prev->pos.y), cv::Point((int)root[i]->pos.x, (int)root[i]->pos.y), color);
			//cvCircle(img, cv::Point((int)prev->pos.x, (int)prev->pos.y), 3, color, -1);
		}

		if (invalid) break;
	}

	//clean up
	for (int i=0; i<n; ++i)
		delete[] ap[i];
	delete[] ap;

	if (radius.size() <= 5)
		return;

	//fclose(f);

	bool isDispersion = false;
	BBOX box;
	//Jun20 simplified determination
	int count = radius.size();
	int midCount = (int)count/2;
	if (radius[0] > radius[count-1] && radius[midCount] > radius[count - 1])
	{
		int r = 0, b = 0;
		int l = 9999, t = 9999;

		for (int i = 0; i < group.size(); i++)
		{
			//cout << group[i]->box.bboxL << " " << group[i]->box.bboxT << " " << group[i]->box.bboxW << " " << group[i]->box.bboxH << endl;
			if (l > group[i]->box.bboxL) l = group[i]->box.bboxL; //cout << l << " ";
			if (t > group[i]->box.bboxT) t = group[i]->box.bboxT; //cout << t << " ";
			if (r < group[i]->box.bboxL + group[i]->box.bboxW) r = group[i]->box.bboxL + group[i]->box.bboxW; //cout << r << " ";
			if (b < group[i]->box.bboxT + group[i]->box.bboxH) b = group[i]->box.bboxT + group[i]->box.bboxH; //cout << b << endl;
		}

		//cout << l << " " << t << " " << r << " " << b << endl;
		box.bboxL = l; box.bboxT = t;
		box.bboxW = r-l; box.bboxH = b-t;

		isDispersion =  true;
	}

	if (isDispersion)
	{
		EventInfo* eve = new EventInfo();
		eve->type = 0;
		eve->endTime = currTime; 
		eve->startTime = RepTrace::timeAddSec(currTime, -60);
		//cout << eve->startTime << " " << currTime << endl;
		for (int j = 0; j < group.size(); j++)
			eve->objIDs.push_back(group[j]->objID);
		eve->reported = false;
		eve->box = box;
		EventType::SetBit(eve->type, dispersionMask);
		events.push_back(eve);
	}
}

void EventDetDLL::DrawObjects(int frameID)
{
	IplImage* img = cvLoadImage(".\\map.bmp");
	//IplImage* img = cvLoadImage(".\\AljuniedMap.png");
	for (int i = 0; i < objects.length; i++)
	{
		RepTrace* trace = objects.GetTrack(i);
		if (trace->outdated || trace->length == 1)
			continue;
		RepPoint* point = trace->GetRoot();
		RepPoint* prevPoint = point;
		//Scalar color = colorList[i%8];
		Scalar color = Scalar(0,0,0);
		int count = trace->length;
		while (count > 0)
		{
			cvCircle(img, cv::Point((int)point->pos.x, (int)point->pos.y), 3, color, -1);
			prevPoint = point;
			point = point->GetPrevPoint();
			if (count >= 2)
				cvLine(img, cvPoint((int)point->pos.x, (int)point->pos.y), cvPoint((int)prevPoint->pos.x, (int)prevPoint->pos.y), color);
			count --;
		}

		cvCircle(img, cv::Point((int)trace->GetRoot()->pos.x, (int)trace->GetRoot()->pos.y), 3, Scalar(0, 0, 0), -1);
	}
	char filename[50];
	sprintf(filename, ".\\output\\tracks_%d.jpg", frameID);
	cvSaveImage(filename, img);
	cvReleaseImage(&img);
}

void EventDetDLL::UpdateStatus()
{
#ifdef DEBUG
	string fname = ".\\eventOutput\\batch" + to_string(processBatchID) +".txt";
	FILE* trackFile = fopen(fname.c_str(), "a");
#endif

	cout << "EventDet Batch " << processBatchID << endl;
	SYSTEMTIME st;
	GetLocalTime(&st);
	cout << "Start:" << st.wHour << ":" << st.wMinute << ":" << st.wSecond << "." << st.wMilliseconds << endl;

	hasEvents = false;
	metaFrame = Scalar(255, 255, 255);
	//metaFrame = imread(".\\AljuniedMap.png", CV_LOAD_IMAGE_COLOR);

	CvFont		font;
	cvInitFont(&font,CV_FONT_VECTOR0,0.6, 0.6, 0.0, 1);//<<<<<<<<<<<<<<<<<<<<<<<<<<<

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
#ifdef DEBUG
		fprintf(trackFile, "%d ", i);
#endif

		if (trace == nullptr)
		{
#ifdef DEBUG
			fprintf(trackFile, "null\n");
#endif
			continue;
		}
		
		if (trace->outdated)
		{
#ifdef DEBUG
			fprintf(trackFile, "outdated\n");
#endif
			continue;
		}

#ifdef DEBUG
		RepPoint* pt = trace->GetRoot();
		fprintf(trackFile, "ID: %d Length %d\n", pt->objID, trace->length);
		while (pt != nullptr)
		{
			fprintf(trackFile,"\t%.3f %.3f %s\n", pt->pos.x, pt->pos.y, pt->ts.c_str());
			pt = pt->GetPrevPoint();
		}
		fprintf(trackFile, "\n\n");
#endif

		trace->UpdateStatus();

		//abnormal loitering
		if (trace->directionChangeCount > 5)
		{
			float loiterTime = RepTrace::GetDuration(trace->GetLastRP(), trace->GetFirstRP());
			if (loiterTime / trace->duration > 0.5)
			{
				EventInfo* eve = new EventInfo();
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
		//cout << "Speed:" << trace->curSpeed << " Acceleration: " << trace->curAcceleration << endl;
		if (trace->curSpeed > RunningSpeedThre && trace->maxAcceleration > RunningAccelerationThre)
		{
			EventInfo* eve = new EventInfo();
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

		//putText(img, currTime.c_str(), cv::Point(10, 30), CV_FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255,0,255), 2.0);

		//string filename = ".\\eventOutput\\" + currTime + ".jpg";
		//imwrite(filename, img);

	}

	FILE* logFile = fopen(".\\eventOutput\\events.txt", "a");
	for (int i = 0; i < events.size(); i++)
	{
		EventInfo* eve = events[i];
		if (EventType::GetBit(eve->type, loiterMask) == 1)
			fprintf(logFile, "Loiter %s %d\n", eve->startTime.c_str(), eve->objIDs[0]);
		else if (EventType::GetBit(eve->type, runningMask) == 1)
			fprintf(logFile, "Running %s %d\n", eve->startTime.c_str(), eve->objIDs[0]);
		else if (EventType::GetBit(eve->type, congregationMask) == 1)
		{
			fprintf(logFile, "Congregation %s ", eve->startTime.c_str());
			for (int j = 0; j < eve->objIDs.size(); j++)
			{
				fprintf(logFile, "%d ", eve->objIDs[j]);
				RepTrace* trace = objects.GetTrackbyID(eve->objIDs[j]);
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
			fprintf(logFile, "\n");
		}
	}

#ifdef DEBUG
	fclose(trackFile);
#endif
	fclose(logFile);

	if (events.size() > 0)
		hasEvents = true;

	GetLocalTime(&st);
	cout << "End:" << st.wHour << ":" << st.wMinute << ":" << st.wSecond << "." << st.wMilliseconds << endl;

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
				if (distToOthers > GroupingDistThre)
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

		_sleep(5000);
	}
}

void EventDetDLL::ReportCongregation(vector<EventInfo*> &eventList)
{
	for (int i = 0; i < events.size(); i++)
	{
		EventInfo* eve = events[i];
		if (EventType::GetBit(eve->type, congregationMask) == 1)
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
			if (RepTrace::GetDistance(obj, track->GetRoot()) >= RPDistThre)
				track->AppendPoint(obj);
			else
			{
				//track->InsertIntrusionStatus(CheckIntrusion(x, y));
				track->UpdateLastTime(obj->ts);
				delete obj;
			}
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
		cout << "EventDet Current Timestamp:" << currTime << endl;
	}
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

	colorList.push_back(Scalar(0, 0, 255));
	colorList.push_back(Scalar(0, 128, 255));
	colorList.push_back(Scalar(0, 255, 255));
	colorList.push_back(Scalar(0, 255, 0));
	colorList.push_back(Scalar(255, 255, 0));
	colorList.push_back(Scalar(255, 0, 0));
	colorList.push_back(Scalar(255, 0, 128));
	colorList.push_back(Scalar(0, 153, 153));
	colorList.push_back(Scalar(0, 0, 0));

	//LoadConfig();
	minGridSize = 200;
	minGroupSize = 3;
	eventMask = imread(".\\data\\EventDet\\eventMask.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	resize(eventMask, eventMask, Size(width, height));
	lastCongregationEvent = new EventInfo();
	lastCongregationEvent->box.bboxW = 0;

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
	//FILE* f = fopen(".\\eventOutput\\append.txt", "a");
	//fprintf(f, "append %d %s (%d, %d, %d, %d)\n", id, ts.c_str(), box.bboxL, box.bboxT, box.bboxW, box.bboxH);
	//fclose(f);
	//cout << "APPEND " << id << " " << box.bboxL << " " << box.bboxT << " " << box.bboxL + box.bboxW << " " << box.bboxT + box.bboxH << endl;
	double value = sum(eventMask(Range(box.bboxT, box.bboxT+box.bboxH-1), Range(box.bboxL, box.bboxL+box.bboxW-1)))[0];
	if (value / (box.bboxW * box.bboxH) < 100)
	{
		//cout << "\tnot added " << id << endl;
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
	//	RepTrace *track = objects.GetTrackbyID(id);
	//	if (track == nullptr)
	//	{
	//		track = new RepTrace();
	//		track->AppendPoint(obj);
	//		track->ID = obj->objID;
	//		objects.AddTrack(track);
	//	}
	//	else
	//	{
	//		if (RepTrace::GetDistance(obj, track->GetRoot()) >= RPDistThre)
	//			track->AppendPoint(obj);
	//		else
	//		{
	//			//track->InsertIntrusionStatus(CheckIntrusion(x, y));
	//			track->UpdateLastTime(ts);
	//			delete obj;
	//		}
	//	}
	//}
}



