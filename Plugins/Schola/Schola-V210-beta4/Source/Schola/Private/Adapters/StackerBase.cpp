// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Adapters/StackerBase.h"
#include "Common/LogSchola.h"
#include "Containers/RingBuffer.h"

void UStackerBase::ValidateDefaultPointAndSpace()
{
	checkf(DefaultPoint.IsValid() && DefaultPoint.GetPtr<FPoint>(), TEXT("UStackerBase::ValidateDefaultPointAndSpace(): DefaultPoint is not a valid FPoint."));
	checkf(UnstackedSpace.IsValid() && UnstackedSpace.GetPtr<FSpace>(), TEXT("UStackerBase::ValidateDefaultPointAndSpace(): UnstackedSpace is not a valid FSpace."));
	checkf(UnstackedSpace.GetPtr<FSpace>()->Contains(DefaultPoint), TEXT("UStackerBase::ValidateDefaultPointAndSpace(): DefaultPoint is not contained in UnstackedSpace."));
}

void UStackerBase::PopulateBufferWithDefaults()
{
	this->ValidateDefaultPointAndSpace();

	const int32 Cap = FMath::Max(1, StackSize);
	Buffer.Reserve(Cap);
	for (int32 i = 0; i < Cap; ++i)
	{
		Buffer.Add(DefaultPoint);
	}
}

void UStackerBase::Push(const FInstancedStruct& InPoint, FInstancedStruct& OutStackedPoint)
{
	if (Buffer.Num() == 0)
	{
		PopulateBufferWithDefaults();
	}
	// Check if the input is valid a valid Point
	if(!InPoint.IsValid() || !InPoint.GetPtr<FPoint>())
	{
		UE_LOG(LogSchola, Error, TEXT("UStackerBase::Push(): InPoint is not a valid FPoint. Ignoring."));
		GetStacked(OutStackedPoint);
		return;
	}
	// Check if the point is contained in the unstacked space.
	if (UnstackedSpace.IsValid())
	{
		const FSpace* Space = UnstackedSpace.GetPtr<FSpace>();
		if (Space && !Space->Contains(reinterpret_cast<const TInstancedStruct<FPoint>&>(InPoint)))
		{
			UE_LOG(LogSchola, Error, TEXT("UStackerBase::Push(): Pushed Point is not contained in UnstackedSpace. Ignoring."));
			GetStacked(OutStackedPoint);
			return;
		}
	}
	const int32 MaxSize = FMath::Max(1, StackSize);
	Buffer.Pop();
	Buffer.AddFront(reinterpret_cast<const TInstancedStruct<FPoint>&>(InPoint));

	NumValid_ = FMath::Min(NumValid_ + 1, MaxSize);
	GetStacked(OutStackedPoint);
}

void UStackerBase::Reset()
{
	Buffer.Reset();
	NumValid_ = 0;
	PopulateBufferWithDefaults();
}
