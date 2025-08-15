#pragma once

class KeyReciever
{
public:
	KeyReciever() = default;
	virtual ~KeyReciever() = default;
	virtual void handleKeyEvents(int key, int action, int scancode, int mods) = 0;
};

class CursorPosReciever
{
public:
	CursorPosReciever() = default;
	virtual ~CursorPosReciever() = default;
	virtual void handleCursorPosEvents(double x, double y) = 0;
};