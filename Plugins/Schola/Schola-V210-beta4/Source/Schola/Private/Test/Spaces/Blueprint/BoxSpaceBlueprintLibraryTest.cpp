// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"
#include "Spaces/Blueprint/BoxSpaceBlueprintLibrary.h"
#include "Spaces/BoxSpace.h"

#if WITH_DEV_AUTOMATION_TESTS

#define TestEqualExactFloat(TestMessage, Actual, Expected) TestEqual(TestMessage, (float)Actual, (float)Expected, 0.0001f)

// ArraysToBoxSpace Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceBlueprintLibrary_ArraysToBoxSpace_BasicTest, "Schola.Spaces.Blueprint.BoxSpaceBlueprintLibrary.ArraysToBoxSpace.Basic", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceBlueprintLibrary_ArraysToBoxSpace_BasicTest::RunTest(const FString& Parameters)
{
    TArray<float> Low = {-1.0f, -2.0f, -3.0f};
    TArray<float> High = {1.0f, 2.0f, 3.0f};
    TArray<int32> Shape = {3};

    TInstancedStruct<FBoxSpace> Result = UBoxSpaceBlueprintLibrary::ArraysToBoxSpace(Low, High, Shape);

    TestTrue(TEXT("Result is valid"), Result.IsValid());
    
    const FBoxSpace& BoxSpace = Result.Get<FBoxSpace>();
    TestEqual(TEXT("BoxSpace.Dimensions.Num() == 3"), BoxSpace.Dimensions.Num(), 3);
    TestEqual(TEXT("BoxSpace.Shape.Num() == 1"), BoxSpace.Shape.Num(), 1);
    TestEqual(TEXT("BoxSpace.Shape[0] == 3"), BoxSpace.Shape[0], 3);

    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[0].Low == -1.0"), BoxSpace.Dimensions[0].Low, -1.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[1].Low == -2.0"), BoxSpace.Dimensions[1].Low, -2.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[2].Low == -3.0"), BoxSpace.Dimensions[2].Low, -3.0f);

    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[0].High == 1.0"), BoxSpace.Dimensions[0].High, 1.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[1].High == 2.0"), BoxSpace.Dimensions[1].High, 2.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[2].High == 3.0"), BoxSpace.Dimensions[2].High, 3.0f);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceBlueprintLibrary_ArraysToBoxSpace_WithShapeTest, "Schola.Spaces.Blueprint.BoxSpaceBlueprintLibrary.ArraysToBoxSpace.WithShape", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceBlueprintLibrary_ArraysToBoxSpace_WithShapeTest::RunTest(const FString& Parameters)
{
    TArray<float> Low = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    TArray<float> High = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    TArray<int32> Shape = {2, 3};

    TInstancedStruct<FBoxSpace> Result = UBoxSpaceBlueprintLibrary::ArraysToBoxSpace(Low, High, Shape);

    TestTrue(TEXT("Result is valid"), Result.IsValid());
    
    const FBoxSpace& BoxSpace = Result.Get<FBoxSpace>();
    TestEqual(TEXT("BoxSpace.Dimensions.Num() == 6"), BoxSpace.Dimensions.Num(), 6);
    TestEqual(TEXT("BoxSpace.Shape.Num() == 2"), BoxSpace.Shape.Num(), 2);
    TestEqual(TEXT("BoxSpace.Shape[0] == 2"), BoxSpace.Shape[0], 2);
    TestEqual(TEXT("BoxSpace.Shape[1] == 3"), BoxSpace.Shape[1], 3);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceBlueprintLibrary_ArraysToBoxSpace_EmptyTest, "Schola.Spaces.Blueprint.BoxSpaceBlueprintLibrary.ArraysToBoxSpace.Empty", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceBlueprintLibrary_ArraysToBoxSpace_EmptyTest::RunTest(const FString& Parameters)
{
    TArray<float> Low;
    TArray<float> High;
    TArray<int32> Shape;

    TInstancedStruct<FBoxSpace> Result = UBoxSpaceBlueprintLibrary::ArraysToBoxSpace(Low, High, Shape);

    TestTrue(TEXT("Result is valid"), Result.IsValid());
    
    const FBoxSpace& BoxSpace = Result.Get<FBoxSpace>();
    TestEqual(TEXT("BoxSpace.Dimensions.Num() == 0"), BoxSpace.Dimensions.Num(), 0);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceBlueprintLibrary_ArraysToBoxSpace_NegativeRangeTest, "Schola.Spaces.Blueprint.BoxSpaceBlueprintLibrary.ArraysToBoxSpace.NegativeRange", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceBlueprintLibrary_ArraysToBoxSpace_NegativeRangeTest::RunTest(const FString& Parameters)
{
    TArray<float> Low = {-10.0f, -20.0f};
    TArray<float> High = {-5.0f, -10.0f};
    TArray<int32> Shape = {2};

    TInstancedStruct<FBoxSpace> Result = UBoxSpaceBlueprintLibrary::ArraysToBoxSpace(Low, High, Shape);

    TestTrue(TEXT("Result is valid"), Result.IsValid());
    
    const FBoxSpace& BoxSpace = Result.Get<FBoxSpace>();
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[0].Low == -10.0"), BoxSpace.Dimensions[0].Low, -10.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[0].High == -5.0"), BoxSpace.Dimensions[0].High, -5.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[1].Low == -20.0"), BoxSpace.Dimensions[1].Low, -20.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[1].High == -10.0"), BoxSpace.Dimensions[1].High, -10.0f);

    return true;
}

// VectorToBoxSpace Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceBlueprintLibrary_VectorToBoxSpace_BasicTest, "Schola.Spaces.Blueprint.BoxSpaceBlueprintLibrary.VectorToBoxSpace.Basic", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceBlueprintLibrary_VectorToBoxSpace_BasicTest::RunTest(const FString& Parameters)
{
    FVector Low(-100.0f, -200.0f, -300.0f);
    FVector High(100.0f, 200.0f, 300.0f);

    TInstancedStruct<FBoxSpace> Result = UBoxSpaceBlueprintLibrary::VectorToBoxSpace(Low, High);

    TestTrue(TEXT("Result is valid"), Result.IsValid());
    
    const FBoxSpace& BoxSpace = Result.Get<FBoxSpace>();
    TestEqual(TEXT("BoxSpace.Dimensions.Num() == 3"), BoxSpace.Dimensions.Num(), 3);
    TestEqual(TEXT("BoxSpace.Shape.Num() == 1"), BoxSpace.Shape.Num(), 1);
    TestEqual(TEXT("BoxSpace.Shape[0] == 3"), BoxSpace.Shape[0], 3);

    // X dimension
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[0].Low == -100.0 (X)"), BoxSpace.Dimensions[0].Low, -100.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[0].High == 100.0 (X)"), BoxSpace.Dimensions[0].High, 100.0f);
    
    // Y dimension
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[1].Low == -200.0 (Y)"), BoxSpace.Dimensions[1].Low, -200.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[1].High == 200.0 (Y)"), BoxSpace.Dimensions[1].High, 200.0f);
    
    // Z dimension
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[2].Low == -300.0 (Z)"), BoxSpace.Dimensions[2].Low, -300.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[2].High == 300.0 (Z)"), BoxSpace.Dimensions[2].High, 300.0f);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceBlueprintLibrary_VectorToBoxSpace_ZeroToOneTest, "Schola.Spaces.Blueprint.BoxSpaceBlueprintLibrary.VectorToBoxSpace.ZeroToOne", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceBlueprintLibrary_VectorToBoxSpace_ZeroToOneTest::RunTest(const FString& Parameters)
{
    FVector Low = FVector::ZeroVector;
    FVector High = FVector::OneVector;

    TInstancedStruct<FBoxSpace> Result = UBoxSpaceBlueprintLibrary::VectorToBoxSpace(Low, High);

    TestTrue(TEXT("Result is valid"), Result.IsValid());
    
    const FBoxSpace& BoxSpace = Result.Get<FBoxSpace>();
    TestEqual(TEXT("BoxSpace.Dimensions.Num() == 3"), BoxSpace.Dimensions.Num(), 3);

    for (int32 i = 0; i < 3; i++)
    {
        TestEqualExactFloat(FString::Printf(TEXT("BoxSpace.Dimensions[%d].Low == 0.0"), i), BoxSpace.Dimensions[i].Low, 0.0f);
        TestEqualExactFloat(FString::Printf(TEXT("BoxSpace.Dimensions[%d].High == 1.0"), i), BoxSpace.Dimensions[i].High, 1.0f);
    }

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceBlueprintLibrary_VectorToBoxSpace_AsymmetricTest, "Schola.Spaces.Blueprint.BoxSpaceBlueprintLibrary.VectorToBoxSpace.Asymmetric", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceBlueprintLibrary_VectorToBoxSpace_AsymmetricTest::RunTest(const FString& Parameters)
{
    FVector Low(-10.5f, 0.0f, -50.0f);
    FVector High(20.5f, 100.0f, 50.0f);

    TInstancedStruct<FBoxSpace> Result = UBoxSpaceBlueprintLibrary::VectorToBoxSpace(Low, High);

    TestTrue(TEXT("Result is valid"), Result.IsValid());
    
    const FBoxSpace& BoxSpace = Result.Get<FBoxSpace>();
    
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[0].Low == -10.5"), BoxSpace.Dimensions[0].Low, -10.5f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[0].High == 20.5"), BoxSpace.Dimensions[0].High, 20.5f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[1].Low == 0.0"), BoxSpace.Dimensions[1].Low, 0.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[1].High == 100.0"), BoxSpace.Dimensions[1].High, 100.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[2].Low == -50.0"), BoxSpace.Dimensions[2].Low, -50.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[2].High == 50.0"), BoxSpace.Dimensions[2].High, 50.0f);

    return true;
}

// RotatorSpace Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceBlueprintLibrary_RotatorSpace_BasicTest, "Schola.Spaces.Blueprint.BoxSpaceBlueprintLibrary.RotatorSpace.Basic", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceBlueprintLibrary_RotatorSpace_BasicTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FBoxSpace> Result = UBoxSpaceBlueprintLibrary::RotatorSpace();

    TestTrue(TEXT("Result is valid"), Result.IsValid());
    
    const FBoxSpace& BoxSpace = Result.Get<FBoxSpace>();
    TestEqual(TEXT("BoxSpace.Dimensions.Num() == 3"), BoxSpace.Dimensions.Num(), 3);
    TestEqual(TEXT("BoxSpace.Shape.Num() == 1"), BoxSpace.Shape.Num(), 1);
    TestEqual(TEXT("BoxSpace.Shape[0] == 3"), BoxSpace.Shape[0], 3);

    // Pitch dimension
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[0].Low == -180.0 (Pitch)"), BoxSpace.Dimensions[0].Low, -180.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[0].High == 180.0 (Pitch)"), BoxSpace.Dimensions[0].High, 180.0f);
    
    // Yaw dimension
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[1].Low == -180.0 (Yaw)"), BoxSpace.Dimensions[1].Low, -180.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[1].High == 180.0 (Yaw)"), BoxSpace.Dimensions[1].High, 180.0f);
    
    // Roll dimension
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[2].Low == -180.0 (Roll)"), BoxSpace.Dimensions[2].Low, -180.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[2].High == 180.0 (Roll)"), BoxSpace.Dimensions[2].High, 180.0f);

    return true;
}

// TransformToBoxSpace Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceBlueprintLibrary_TransformToBoxSpace_BasicTest, "Schola.Spaces.Blueprint.BoxSpaceBlueprintLibrary.TransformToBoxSpace.Basic", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceBlueprintLibrary_TransformToBoxSpace_BasicTest::RunTest(const FString& Parameters)
{
    FVector LocationLow(-100.0f, -100.0f, -100.0f);
    FVector LocationHigh(100.0f, 100.0f, 100.0f);
    FVector ScaleLow(0.1f, 0.1f, 0.1f);
    FVector ScaleHigh(10.0f, 10.0f, 10.0f);

    TInstancedStruct<FBoxSpace> Result = UBoxSpaceBlueprintLibrary::TransformToBoxSpace(
        LocationLow, LocationHigh,
        ScaleLow, ScaleHigh
    );

    TestTrue(TEXT("Result is valid"), Result.IsValid());
    
    const FBoxSpace& BoxSpace = Result.Get<FBoxSpace>();
    TestEqual(TEXT("BoxSpace.Dimensions.Num() == 9"), BoxSpace.Dimensions.Num(), 9);
    TestEqual(TEXT("BoxSpace.Shape.Num() == 1"), BoxSpace.Shape.Num(), 1);
    TestEqual(TEXT("BoxSpace.Shape[0] == 9"), BoxSpace.Shape[0], 9);

    // Location dimensions (0-2)
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[0].Low == -100.0 (Location.X)"), BoxSpace.Dimensions[0].Low, -100.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[0].High == 100.0 (Location.X)"), BoxSpace.Dimensions[0].High, 100.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[1].Low == -100.0 (Location.Y)"), BoxSpace.Dimensions[1].Low, -100.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[1].High == 100.0 (Location.Y)"), BoxSpace.Dimensions[1].High, 100.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[2].Low == -100.0 (Location.Z)"), BoxSpace.Dimensions[2].Low, -100.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[2].High == 100.0 (Location.Z)"), BoxSpace.Dimensions[2].High, 100.0f);

    // Rotation dimensions (3-5) - always [-180, 180]
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[3].Low == -180.0 (Rotation.Pitch)"), BoxSpace.Dimensions[3].Low, -180.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[3].High == 180.0 (Rotation.Pitch)"), BoxSpace.Dimensions[3].High, 180.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[4].Low == -180.0 (Rotation.Yaw)"), BoxSpace.Dimensions[4].Low, -180.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[4].High == 180.0 (Rotation.Yaw)"), BoxSpace.Dimensions[4].High, 180.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[5].Low == -180.0 (Rotation.Roll)"), BoxSpace.Dimensions[5].Low, -180.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[5].High == 180.0 (Rotation.Roll)"), BoxSpace.Dimensions[5].High, 180.0f);

    // Scale dimensions (6-8)
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[6].Low == 0.1 (Scale.X)"), BoxSpace.Dimensions[6].Low, 0.1f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[6].High == 10.0 (Scale.X)"), BoxSpace.Dimensions[6].High, 10.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[7].Low == 0.1 (Scale.Y)"), BoxSpace.Dimensions[7].Low, 0.1f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[7].High == 10.0 (Scale.Y)"), BoxSpace.Dimensions[7].High, 10.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[8].Low == 0.1 (Scale.Z)"), BoxSpace.Dimensions[8].Low, 0.1f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[8].High == 10.0 (Scale.Z)"), BoxSpace.Dimensions[8].High, 10.0f);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceBlueprintLibrary_TransformToBoxSpace_IdentityRangeTest, "Schola.Spaces.Blueprint.BoxSpaceBlueprintLibrary.TransformToBoxSpace.IdentityRange", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceBlueprintLibrary_TransformToBoxSpace_IdentityRangeTest::RunTest(const FString& Parameters)
{
    FVector LocationLow = FVector::ZeroVector;
    FVector LocationHigh = FVector::ZeroVector;
    FVector ScaleLow = FVector::OneVector;
    FVector ScaleHigh = FVector::OneVector;

    TInstancedStruct<FBoxSpace> Result = UBoxSpaceBlueprintLibrary::TransformToBoxSpace(
        LocationLow, LocationHigh,
        ScaleLow, ScaleHigh
    );

    TestTrue(TEXT("Result is valid"), Result.IsValid());
    
    const FBoxSpace& BoxSpace = Result.Get<FBoxSpace>();
    TestEqual(TEXT("BoxSpace.Dimensions.Num() == 9"), BoxSpace.Dimensions.Num(), 9);

    // Location should be zero
    for (int32 i = 0; i < 3; i++)
    {
        TestEqualExactFloat(FString::Printf(TEXT("BoxSpace.Dimensions[%d].Low == 0.0 (Location)"), i), BoxSpace.Dimensions[i].Low, 0.0f);
        TestEqualExactFloat(FString::Printf(TEXT("BoxSpace.Dimensions[%d].High == 0.0 (Location)"), i), BoxSpace.Dimensions[i].High, 0.0f);
    }

    // Rotation should always be [-180, 180]
    for (int32 i = 3; i < 6; i++)
    {
        TestEqualExactFloat(FString::Printf(TEXT("BoxSpace.Dimensions[%d].Low == -180.0 (Rotation)"), i), BoxSpace.Dimensions[i].Low, -180.0f);
        TestEqualExactFloat(FString::Printf(TEXT("BoxSpace.Dimensions[%d].High == 180.0 (Rotation)"), i), BoxSpace.Dimensions[i].High, 180.0f);
    }

    // Scale should be one
    for (int32 i = 6; i < 9; i++)
    {
        TestEqualExactFloat(FString::Printf(TEXT("BoxSpace.Dimensions[%d].Low == 1.0 (Scale)"), i), BoxSpace.Dimensions[i].Low, 1.0f);
        TestEqualExactFloat(FString::Printf(TEXT("BoxSpace.Dimensions[%d].High == 1.0 (Scale)"), i), BoxSpace.Dimensions[i].High, 1.0f);
    }

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceBlueprintLibrary_TransformToBoxSpace_AsymmetricTest, "Schola.Spaces.Blueprint.BoxSpaceBlueprintLibrary.TransformToBoxSpace.Asymmetric", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceBlueprintLibrary_TransformToBoxSpace_AsymmetricTest::RunTest(const FString& Parameters)
{
    FVector LocationLow(-50.0f, 0.0f, -100.0f);
    FVector LocationHigh(150.0f, 200.0f, 100.0f);
    FVector ScaleLow(0.5f, 0.5f, 0.5f);
    FVector ScaleHigh(2.0f, 3.0f, 4.0f);

    TInstancedStruct<FBoxSpace> Result = UBoxSpaceBlueprintLibrary::TransformToBoxSpace(
        LocationLow, LocationHigh,
        ScaleLow, ScaleHigh
    );

    TestTrue(TEXT("Result is valid"), Result.IsValid());
    
    const FBoxSpace& BoxSpace = Result.Get<FBoxSpace>();
    
    // Location
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[0].Low == -50.0"), BoxSpace.Dimensions[0].Low, -50.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[0].High == 150.0"), BoxSpace.Dimensions[0].High, 150.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[1].Low == 0.0"), BoxSpace.Dimensions[1].Low, 0.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[1].High == 200.0"), BoxSpace.Dimensions[1].High, 200.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[2].Low == -100.0"), BoxSpace.Dimensions[2].Low, -100.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[2].High == 100.0"), BoxSpace.Dimensions[2].High, 100.0f);

    // Rotation - always [-180, 180]
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[3].Low == -180.0"), BoxSpace.Dimensions[3].Low, -180.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[3].High == 180.0"), BoxSpace.Dimensions[3].High, 180.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[4].Low == -180.0"), BoxSpace.Dimensions[4].Low, -180.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[4].High == 180.0"), BoxSpace.Dimensions[4].High, 180.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[5].Low == -180.0"), BoxSpace.Dimensions[5].Low, -180.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[5].High == 180.0"), BoxSpace.Dimensions[5].High, 180.0f);

    // Scale
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[6].Low == 0.5"), BoxSpace.Dimensions[6].Low, 0.5f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[6].High == 2.0"), BoxSpace.Dimensions[6].High, 2.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[7].Low == 0.5"), BoxSpace.Dimensions[7].Low, 0.5f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[7].High == 3.0"), BoxSpace.Dimensions[7].High, 3.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[8].Low == 0.5"), BoxSpace.Dimensions[8].Low, 0.5f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[8].High == 4.0"), BoxSpace.Dimensions[8].High, 4.0f);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceBlueprintLibrary_TransformToBoxSpace_NegativeScaleTest, "Schola.Spaces.Blueprint.BoxSpaceBlueprintLibrary.TransformToBoxSpace.NegativeScale", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceBlueprintLibrary_TransformToBoxSpace_NegativeScaleTest::RunTest(const FString& Parameters)
{
    FVector LocationLow = FVector::ZeroVector;
    FVector LocationHigh = FVector::ZeroVector;
    FVector ScaleLow(-2.0f, -2.0f, -2.0f);
    FVector ScaleHigh(2.0f, 2.0f, 2.0f);

    TInstancedStruct<FBoxSpace> Result = UBoxSpaceBlueprintLibrary::TransformToBoxSpace(
        LocationLow, LocationHigh,
        ScaleLow, ScaleHigh
    );

    TestTrue(TEXT("Result is valid"), Result.IsValid());
    
    const FBoxSpace& BoxSpace = Result.Get<FBoxSpace>();
    
    // Scale dimensions should support negative values
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[6].Low == -2.0"), BoxSpace.Dimensions[6].Low, -2.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[6].High == 2.0"), BoxSpace.Dimensions[6].High, 2.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[7].Low == -2.0"), BoxSpace.Dimensions[7].Low, -2.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[7].High == 2.0"), BoxSpace.Dimensions[7].High, 2.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[8].Low == -2.0"), BoxSpace.Dimensions[8].Low, -2.0f);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[8].High == 2.0"), BoxSpace.Dimensions[8].High, 2.0f);

    return true;
}

#endif




