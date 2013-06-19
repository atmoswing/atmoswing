/** 
 *
 *  This file is part of the AtmoSwing software.
 *
 *  Copyright (c) 2008-2012  University of Lausanne, Pascal Horton (pascal.horton@unil.ch). 
 *  All rights reserved.
 *
 *  THIS CODE, SOFTWARE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY  
 *  OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR
 *  PURPOSE.
 *
 */
 
#ifndef ASDATAPREDICTANDPRECIPITATION_H
#define ASDATAPREDICTANDPRECIPITATION_H

#include <asIncludes.h>
#include <asDataPredictand.h>


class asDataPredictandPrecipitation: public asDataPredictand
{
public:
    asDataPredictandPrecipitation(PredictandDB predictandDB);
    virtual ~asDataPredictandPrecipitation();

    virtual bool Load(const wxString &AlternateFilePath = wxEmptyString);

    virtual bool Save(const wxString &AlternateDestinationDir = wxEmptyString);

    virtual bool BuildPredictandDB(const wxString &AlternateCatalogFilePath = wxEmptyString, const wxString &AlternateDataDir = wxEmptyString, const wxString &AlternatePatternDir = wxEmptyString, const wxString &AlternateDestinationDir = wxEmptyString);

    virtual Array1DFloat &GetReferenceAxis()
    {
        return m_ReturnPeriods;
    }

    virtual float GetReferenceValue(int i_station, double duration, float reference)
    {
        return GetPrecipitationOfReturnPeriod(i_station, duration, reference);
    }

    virtual Array2DFloat &GetReferenceValuesArray()
    {
        return m_DailyPrecipitationsForReturnPeriods;
    }

    float GetPrecipitationOfReturnPeriod(int i_station, double duration, float returnPeriod);

	void SetReturnPeriodNormalization(float val)
	{
		m_ReturnPeriodNormalization = val;
	}

	float GetReturnPeriodNormalization()
	{
		return m_ReturnPeriodNormalization;
	}
	
	void SetIsSqrt(bool val)
	{
		m_IsSqrt = val;
	}

	bool IsSqrt()
	{
		return m_IsSqrt;
	}


protected:

private:

    float m_ReturnPeriodNormalization;
	bool m_IsSqrt;
    // Vector (dim = return periods)
    Array1DFloat m_ReturnPeriods;
    // Matrix data
    Array2DFloat m_GumbelDuration; // Values of the Precipitation duration
    Array2DFloat m_GumbelParamA; // Values of the a parameter of the Gumbel adjustment
    Array2DFloat m_GumbelParamB; // Values of the b parameter of the Gumbel adjustment
    // Matrix with other axes
    Array2DFloat m_DailyPrecipitationsForReturnPeriods;

    /** Initialize the containers
     * \return True on success
     */
    bool InitContainers();

    /** Process a Gumbel adjustemnt by means of the moments method
     * \return True on success
     * \link http://echo2.epfl.ch/e-drologie/chapitres/annexes/AnalFrequ_rep.html#moments
     */
    bool MakeGumbelAdjustment();

    bool BuildDataNormalized();

    bool BuildDailyRPV10();

    bool BuildDailyPrecipitationsForAllReturnPeriods();

};

#endif // ASDATAPREDICTANDPRECIPITATION_H
