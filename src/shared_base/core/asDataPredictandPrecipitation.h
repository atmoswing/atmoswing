/*
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS HEADER.
 *
 * The contents of this file are subject to the terms of the
 * Common Development and Distribution License (the "License").
 * You may not use this file except in compliance with the License.
 *
 * You can read the License at http://opensource.org/licenses/CDDL-1.0
 * See the License for the specific language governing permissions
 * and limitations under the License.
 * 
 * When distributing Covered Code, include this CDDL Header Notice in 
 * each file and include the License file (licence.txt). If applicable, 
 * add the following below this CDDL Header, with the fields enclosed
 * by brackets [] replaced by your own identifying information:
 * "Portions Copyright [year] [name of copyright owner]"
 * 
 * The Original Software is AtmoSwing.
 * The Original Software was developed at the University of Lausanne.
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2008-2013 Pascal Horton, University of Lausanne.
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 */

#ifndef ASDATAPREDICTANDPRECIPITATION_H
#define ASDATAPREDICTANDPRECIPITATION_H

#include <asIncludes.h>
#include <asDataPredictand.h>


class asDataPredictandPrecipitation
        : public asDataPredictand
{
public:
    asDataPredictandPrecipitation(Parameter dataParameter, TemporalResolution dataTemporalResolution,
                                  SpatialAggregation dataSpatialAggregation);

    virtual ~asDataPredictandPrecipitation();

    virtual bool Load(const wxString &filePath);

    virtual bool Save(const wxString &AlternateDestinationDir = wxEmptyString) const;

    virtual bool BuildPredictandDB(const wxString &catalogFilePath, const wxString &AlternateDataDir = wxEmptyString,
                                   const wxString &AlternatePatternDir = wxEmptyString,
                                   const wxString &AlternateDestinationDir = wxEmptyString);

    virtual Array1DFloat GetReferenceAxis() const
    {
        return m_returnPeriods;
    }

    virtual float GetReferenceValue(int iStat, double duration, float reference) const
    {
        return GetPrecipitationOfReturnPeriod(iStat, duration, reference);
    }

    virtual Array2DFloat GetReferenceValuesArray() const
    {
        return m_dailyPrecipitationsForReturnPeriods;
    }

    float GetPrecipitationOfReturnPeriod(int iStat, double duration, float returnPeriod) const;

    void SetReturnPeriodNormalization(float val)
    {
        m_returnPeriodNormalization = val;
    }

    float GetReturnPeriodNormalization() const
    {
        return m_returnPeriodNormalization;
    }

    void SetIsSqrt(bool val)
    {
        m_isSqrt = val;
    }

    bool IsSqrt() const
    {
        return m_isSqrt;
    }


protected:

private:
    float m_returnPeriodNormalization;
    bool m_isSqrt;
    // Vector (dim = return periods)
    Array1DFloat m_returnPeriods;
    // Matrix data
    Array2DFloat m_gumbelDuration; // Values of the Precipitation duration
    Array2DFloat m_gumbelParamA; // Values of the a parameter of the Gumbel adjustment
    Array2DFloat m_gumbelParamB; // Values of the b parameter of the Gumbel adjustment
    // Matrix with other axes
    Array2DFloat m_dailyPrecipitationsForReturnPeriods;

    bool InitContainers();

    bool MakeGumbelAdjustment();

    bool BuildDataNormalized();

    bool BuildDailyPrecipitationsForAllReturnPeriods();

};

#endif // ASDATAPREDICTANDPRECIPITATION_H
