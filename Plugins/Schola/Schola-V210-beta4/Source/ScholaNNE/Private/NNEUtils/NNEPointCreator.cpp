// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "NNEUtils/NNEPointCreator.h"
#include "LogScholaNNE.h"


int GetMaxIndex(TConstArrayView<float>& InArray)
{
	float CurrMax = InArray[0];
	int	  Index = 0;
	int	  CurrIndex = 0;
	for (const float& Value : InArray)
	{
		if (Value > CurrMax)
		{
			CurrMax = Value;
			Index = CurrIndex;
		}
		CurrIndex += 1;
	}
	return Index;
}


void FNNEPointCreator::operator()(const FNNEDictBuffer& InBuffer)
{
	if (!this->OutputPoint.IsValid())
	{
		this->OutputPoint.InitializeAs<FDictPoint>();
	}

    for(const TPair<FString, TInstancedStruct<FNNEPointBuffer>>& Pair : InBuffer.Buffers)
    {
        const FString& Key = Pair.Key;
		const TInstancedStruct<FNNEPointBuffer>& Buffer = Pair.Value;
        if (this->Space.Get<FDictSpace>().Spaces.Contains(Key))
        {
            // Use const references to avoid unnecessary copies during cleanup
            const TInstancedStruct<FSpace>& SubSpace = this->Space.Get<FDictSpace>().Spaces[Key];
			//This might not exist so we use FindOrAdd
			TInstancedStruct<FPoint>&		SubPoint = this->OutputPoint.GetMutable<FDictPoint>().Points.FindOrAdd(Key);
			FNNEPointCreator::CreatePoint(Buffer, SubPoint, SubSpace);
        }
        else
        {
			UE_LOGFMT(LogScholaNNE, Error, "FNNEPointCreator::operator(): Key {0} not found in Dict Space", Key);
        }
    }
}

void FNNEPointCreator::operator()(const FNNEBoxBuffer& InBuffer)
{
	if (!this->OutputPoint.IsValid())
	{
		this->OutputPoint.InitializeAs<FBoxPoint>();
	}

    this->OutputPoint.GetMutable<FBoxPoint>().Values = InBuffer.Buffer;
}

void FNNEPointCreator::operator()(const FNNEMultiBinaryBuffer& InBuffer)
{
	if (!this->OutputPoint.IsValid())
	{
		this->OutputPoint.InitializeAs<FMultiBinaryPoint>();
	}

	this->OutputPoint.GetMutable<FMultiBinaryPoint>().Values = InBuffer.Buffer;
}

void FNNEPointCreator::operator()(const FNNEDiscreteBuffer& InBuffer)
{
	if (!this->OutputPoint.IsValid())
	{
		this->OutputPoint.InitializeAs<FDiscretePoint>();
	}
	this->OutputPoint.GetMutable<FDiscretePoint>().Value = InBuffer.Buffer[0];
}

void FNNEPointCreator::operator()(const FNNEMultiDiscreteBuffer& InBuffer)
{
	if (!this->OutputPoint.IsValid())
	{
		this->OutputPoint.InitializeAs<FMultiDiscretePoint>();
	}

	FMultiDiscretePoint&	   OutputPointRef = this->OutputPoint.GetMutable<FMultiDiscretePoint>();
	OutputPointRef.Values = InBuffer.Buffer;
}

