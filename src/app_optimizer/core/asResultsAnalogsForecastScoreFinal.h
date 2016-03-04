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
 * Portions Copyright 2013 Pascal Horton, Terr@num.
 */
 
#ifndef ASRESULTSANALOGSFORECASTSCOREFINAL_H
#define ASRESULTSANALOGSFORECASTSCOREFINAL_H

#include <asIncludes.h>
#include <asResults.h>

class asParametersScoring;

class asResultsAnalogsForecastScoreFinal: public asResults
{
public:

    /** Default constructor */
    asResultsAnalogsForecastScoreFinal();

    /** Default destructor */
    virtual ~asResultsAnalogsForecastScoreFinal();

    /** Init
     * \param params The parameters structure
     */
    void Init(asParametersScoring &params);

    /** Access m_forecastScore
     * \return The value of m_forecastScore
     */
    float GetForecastScore()
    {
        return m_forecastScore;
    }

    /** Set m_forecastScore
     * \param val The new value to set
     */
    void SetForecastScore(float val)
    {
        m_forecastScore = val;
    }
    
    /** Access m_forecastScoreArray
     * \return The value of m_forecastScoreArray
     */
    Array1DFloat GetForecastScoreArray()
    {
        return m_forecastScoreArray;
    }

    /** Set m_forecastScoreArray
     * \param val The new value to set
     */
    void SetForecastScore(Array1DFloat val)
    {
        m_forecastScoreArray = val;
        m_hasSingleValue = false;
    }

    /** Set m_forecastScoreArray
     * \param val The new value to set
     */
    void SetForecastScoreArray(Array1DFloat val)
    {
        m_forecastScoreArray = val;
        m_hasSingleValue = false;
    }

    /** Save the result file
     * \param AlternateFilePath An optional file path
     * \return True on success
     */
    bool Save(const wxString &AlternateFilePath = wxEmptyString);

    /** Load the result file
     * \param AlternateFilePath An optional file path
     * \return True on success
     */
    bool Load(const wxString &AlternateFilePath = wxEmptyString);

protected:

    /** Build the result file path
     * \param params The parameters structure
     */
    void BuildFileName(asParametersScoring &params);

private:
    bool m_hasSingleValue;
    float m_forecastScore; //!< Member variable "m_forecastScore"
    Array1DFloat m_forecastScoreArray; //!< Member variable "m_forecastScoreArray"
};

#endif // ASRESULTSANALOGSFORECASTSCOREFINAL_H
