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
 * The Original Software is AtmoSwing. The Initial Developer of the 
 * Original Software is Pascal Horton of the University of Lausanne. 
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2008-2013 University of Lausanne.
 */
 
#ifndef ASDATAPREDICTANDPRECIPITATION_H
#define ASDATAPREDICTANDPRECIPITATION_H

#include <asIncludes.h>
#include <asDataPredictand.h>


class asDataPredictandPrecipitation: public asDataPredictand
{
public:
    asDataPredictandPrecipitation(DataParameter dataParameter, DataTemporalResolution dataTemporalResolution, DataSpatialAggregation dataSpatialAggregation);
    virtual ~asDataPredictandPrecipitation();

    virtual bool Load(const wxString &filePath);

    virtual bool Save(const wxString &AlternateDestinationDir = wxEmptyString);

    virtual bool BuildPredictandDB(const wxString &catalogFilePath, const wxString &AlternateDataDir = wxEmptyString, const wxString &AlternatePatternDir = wxEmptyString, const wxString &AlternateDestinationDir = wxEmptyString);

    virtual Array1DFloat GetReferenceAxis()
    {
        return m_returnPeriods;
    }

    virtual float GetReferenceValue(int i_station, double duration, float reference)
    {
        return GetPrecipitationOfReturnPeriod(i_station, duration, reference);
    }

    virtual Array2DFloat GetReferenceValuesArray()
    {
        return m_dailyPrecipitationsForReturnPeriods;
    }

    float GetPrecipitationOfReturnPeriod(int i_station, double duration, float returnPeriod);

    void SetReturnPeriodNormalization(float val)
    {
        m_returnPeriodNormalization = val;
    }

    float GetReturnPeriodNormalization()
    {
        return m_returnPeriodNormalization;
    }
    
    void SetIsSqrt(bool val)
    {
        m_isSqrt = val;
    }

    bool IsSqrt()
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
