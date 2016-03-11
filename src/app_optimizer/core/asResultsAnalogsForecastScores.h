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

    /** Access m_targetDates
     * \return The whole array m_targetDates
     */
    Array1DFloat &GetTargetDates()
    {
        return m_targetDates;
    }

    /** Set m_targetDates
     * \param refDates The new array to set
     */
    void SetTargetDates(Array1DDouble &refDates)
    {
        m_targetDates.resize(refDates.rows());
        for (int i=0; i<refDates.size(); i++)
        {
            m_targetDates[i] = (float)refDates[i];
            wxASSERT_MSG(m_targetDates[i]>1,_("The target time array has unconsistent values"));
        }
    }

    /** Set m_targetDates
     * \param refDates The new array to set
     */
    void SetTargetDates(Array1DFloat &refDates)
    {
        m_targetDates.resize(refDates.rows());
        m_targetDates = refDates;
    }

    /** Access m_forecastScores
     * \return The whole array m_forecastScores
     */
    Array1DFloat &GetForecastScores()
    {
        return m_forecastScores;
    }

    /** Access m_forecastScores2DArray
     * \return The whole array m_forecastScores2DArray
     */
    Array2DFloat &GetForecastScores2DArray()
    {
        return m_forecastScores2DArray;
    }

    /** Set m_forecastScores
     * \param forecastScores The new array to set
     */
    void SetForecastScores(Array1DDouble &forecastScores)
    {
        m_forecastScores.resize(forecastScores.rows());
        for (int i=0; i<forecastScores.size(); i++)
        {
            m_forecastScores[i] = (float)forecastScores[i];
        }
    }

    /** Set m_forecastScores
     * \param forecastScores The new array to set
     */
    void SetForecastScores(Array1DFloat &forecastScores)
    {
        m_forecastScores.resize(forecastScores.rows());
        m_forecastScores = forecastScores;
    }

    /** Set m_forecastScores
     * \param forecastScores The new array to set
     */
    void SetForecastScores2DArray(Array2DFloat &forecastScores)
    {
        m_forecastScores2DArray.resize(forecastScores.rows(),forecastScores.cols());
        m_forecastScores2DArray = forecastScores;
    }

    /** Get the length of the target time dimension
     * \return The length of the target time
     */
    int GetTargetDatesLength()
    {
        return m_targetDates.size();
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
    Array1DFloat m_targetDates; //!< Member variable "m_targetDates"
    Array1DFloat m_forecastScores; //!< Member variable "m_forecastScores"
    Array2DFloat m_forecastScores2DArray; //!< Member variable "m_forecastScores2DArray"
};

#endif // ASRESULTSANALOGSFORECASTSCORES_H
