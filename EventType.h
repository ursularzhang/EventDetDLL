#ifndef EVENTTYPE_H_
#define EVENTTYPE_H_


namespace EventDetection
{
	static int loiterMask = 1;
	static int abLoiterMask = 2;
	static int runningMask = 4;
	static int intrusionMask = 8;
	static int congregationMask = 16;
	static int dispersionMask = 32;
	static int abandonedobjMask = 64;

	class EventType
	{

		public:
			EventType();
			static int GetBit(int input, int mask);
			static void SetBit(int &input, int mask);
			static void ClearBit(int &input, int mask);
	};
}

#endif