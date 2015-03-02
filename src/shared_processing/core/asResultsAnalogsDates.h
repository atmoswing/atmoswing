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
 
#ifndef ASRESULTSANALOGSDATES_H
#define ASRESULTSANALOGSDATES_H

#include <asIncludes.h>
#include <asResults.h>

class asResultsAnalogsDates: public asResults
{
public:

    /** Default constructor */
    asResultsAnalogsDates();

    /** Default destructor */
    virtual ~asResultsAnalogsDates();

    /** Init
     * \param params The parameters structure
     */
    void Init(asParameters &params);

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

    /** Access m_analogCriteria
     * \return The whole array m_analogCriteria
     */
    Array2DFloat &GetAnalogsCriteria()
    {
        return m_analogsCriteria;
    }

    /** Set m_analogCriteria
     * \param analogCriteria The new array to set
     */
    void SetAnalogsCriteria(Array2DFloat &analogsCriteria)
    {
        m_analogsCriteria.resize(analogsCriteria.rows(), analogsCriteria.cols());
        m_analogsCriteria = analogsCriteria;
    }

    /** Access m_analogDates
     * \return The whole array m_analogDates
     */
    Array2DFloat &GetAnalogsDates()
    {
        return m_analogsDates;
    }

    /** Set m_analogDates
     * \param analogDates The new array to set
     */
    void SetAnalogsDates(Array2DFloat &analogsDates)
    {
        m_analogsDates.resize(analogsDates.rows(), analogsDates.cols());
        m_analogsDates = analogsDates;
    }

    /** Get the length of the target time dimension
     * \return The length of the target time
     */
    int GetTargetDatesLength()
    {
        return m_targetDates.size();
    }

    /** Get the length of the analogs dimension
     * \return The length of the analogs
     */
    int GetAnalogsDatesLength()
    {
        return m_analogsDates.cols();
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
    void BuildFileName();

private:
    Array1DFloat m_targetDates; //!< Member variable "m_targetDates"
    Array2DFloat m_analogsCriteria; //!< Member variable "m_analogCriteria"
    Array2DFloat m_analogsDates; //!< Member variable "m_analogDates"

};

#endif // ASRESULTSANALOGSDATES_H
