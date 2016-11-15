#pragma once
#include "Stdafx.h"
#include "RepPoint.h"
#include <vector>

using namespace EventDetection;
using namespace std;

class TrackList{
	public:
		TrackList();
		~TrackList();
		int length;
	private:
		std::vector<RepTrace*> glbObjects;
	public:
		void AddTrack(RepTrace* track);
		void DeleteTrack(int idx);
		void CombineTrack(RepTrace* track);
		RepTrace* GetTrack(int glbObjIdx);
		void ClearTracks();
		RepTrace* GetTrackbyID(int localID);
};