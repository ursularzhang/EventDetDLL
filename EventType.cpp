#include "Stdafx.h"
#include "EventType.h"

using namespace EventDetection;

EventType::EventType()
{
}

void EventType::ClearBit(int &input, int mask)
{
	input &= ~(mask);
}

void EventType::SetBit(int &input, int mask)
{
	input |= mask;
}

int EventType::GetBit(int input, int mask)
{
	if (input & mask) 
		return 1;
	else 
		return 0;
}