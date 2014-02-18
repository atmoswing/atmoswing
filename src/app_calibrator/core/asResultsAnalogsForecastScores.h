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
 
#ifndef ASRESULTSANALOGSFORECASTSCORES_H
#define ASRESULTSANALOGSFORECASTSCORES_H

#include <asIncludes.h>
#include <asResults.h>

class asParametersScoring;

class asResultsAnalogsForecastScores: public asResults
{
public:

    /** Default constructor */
    asResultsAnalogsForecastScores();

    /** Default destructor */
    virtual ~asResultsAnalogsForecastScores();

    /** Init
     * \param params The parameters structure
     */
    void Init(asParametersScoring &params);

    /** Access m_TargetDates
     * \return The whole array m_TargetDates
     */
    Array1DFloat &GetTargetDates()
    {
        return m_TargetDates;
    }

    /** Set m_TargetDates
     * \param refDates The new array to set
     */
    void SetTargetDates(Array1DDouble &refDates)
    {
        m_TargetDates.resize(refDates.rows());
        for (int i=0; i<refDates.size(); i++)
        {
            m_TargetDates[i] = (float)refDates[i];
            wxASSERT_MSG(m_TargetDates[i]>1,_("The target time array has unconsistent values"));
        }
    }

    /** Set m_TargetDates
     * \param refDates The new array to set
     */
    void SetTargetDates(Array1DFloat &refDates)
    {
        m_TargetDates.resize(refDates.rows());
        m_TargetDates = refDates;
    }

    /** Access m_ForecastScores
     * \return The whole array m_ForecastScores
     */
    Array1DFloat &GetForecastScores()
    {
        return m_ForecastScores;
    }

    /** Access m_ForecastScores2DArray
     * \return The whole array m_ForecastScores2DArray
     */
    Array2DFloat &GetForecastScores2DArray()
    {
        return m_ForecastScores2DArray;
    }

    /** Set m_ForecastScores
     * \param forecastScores The new array to set
     */
    void SetForecastScores(Array1DDouble &forecastScores)
    {
        m_ForecastScores.resize(forecastScores.rows());
        for (int i=0; i<forecastScores.size(); i++)
        {
            m_ForecastScores[i] = (float)forecastScores[i];
        }
    }

    /** Set m_ForecastScores
     * \param forecastScores The new array to set
     */
    void SetForecastScores(Array1DFloat &forecastScores)
    {
        m_ForecastScores.resize(forecastScores.rows());
        m_ForecastScores = forecastScores;
    }

    /** Set m_ForecastScores
     * \param forecastScores The new array to set
     */
    void SetForecastScores2DArray(Array2DFloat &forecastScores)
    {
        m_ForecastScores2DArray.resize(forecastScores.rows(),forecastScores.cols());
        m_ForecastScores2DArray = forecastScores;
    }

    /** Get the length of the target time dimension
     * \return The length of the target time
     */
    int GetTargetDatesLength()
    {
        return m_TargetDates.size();
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
    Array1DFloat m_TargetDates; //!< Member variable "m_TargetDates"
    Array1DFloat m_ForecastScores; //!< Member variable "m_ForecastScores"
    Array2DFloat m_ForecastScores2DArray; //!< Member variable "m_ForecastScores2DArray"
};

#endif // ASRESULTSANALOGSFORECASTSCORES_H
