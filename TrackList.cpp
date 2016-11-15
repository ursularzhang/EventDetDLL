#include "Stdafx.h"
#include "TrackList.h"

using namespace EventDetection;

TrackList::TrackList(){
	length = 0;
}
TrackList::~TrackList(){
}
RepTrace* TrackList::GetTrack(int glbObjIdx)
{
	return glbObjects[glbObjIdx];
}

RepTrace* TrackList::GetTrackbyID(int localID)
{
	RepTrace* track;
	int i;
	for (i = 0; i < glbObjects.size(); i++)
	{
		track = glbObjects[i];
		if (track == nullptr)
			continue;
		if (track->ID == localID)
			return track;
	}
	if (i == glbObjects.size())
		return nullptr;
}

void TrackList::CombineTrack(RepTrace* track)
{
	for (int i = 0; i < glbObjects.size(); i++)
	{
		if (track->ID == (*glbObjects[i]).ID)
		{
			//combine track
			RepPoint* obj = track->GetFirstPoint();
			obj->SetPrevPoint((*glbObjects[i]).GetRoot());
			(*glbObjects[i]).SetRoot(track->GetRoot());//correct?
		}
	}
}
	
void TrackList::AddTrack(RepTrace* track)
{
	glbObjects.push_back(track);
	length++;
}

void TrackList::DeleteTrack(int idx)
{
	RepTrace* track = glbObjects[idx];
	delete track;

	glbObjects.erase(glbObjects.begin()+idx);
	length--;
}

void TrackList::ClearTracks()
{
	glbObjects.clear();
	length = 0;
}
