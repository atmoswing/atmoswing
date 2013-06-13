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

    /** Access m_AnalogCriteria
     * \return The whole array m_AnalogCriteria
     */
    Array2DFloat &GetAnalogsCriteria()
    {
        return m_AnalogsCriteria;
    }

    /** Set m_AnalogCriteria
     * \param analogCriteria The new array to set
     */
    void SetAnalogsCriteria(Array2DFloat &analogsCriteria)
    {
        m_AnalogsCriteria.resize(analogsCriteria.rows(), analogsCriteria.cols());
        m_AnalogsCriteria = analogsCriteria;
    }

    /** Access m_AnalogDates
     * \return The whole array m_AnalogDates
     */
    Array2DFloat &GetAnalogsDates()
    {
        return m_AnalogsDates;
    }

    /** Set m_AnalogDates
     * \param analogDates The new array to set
     */
    void SetAnalogsDates(Array2DFloat &analogsDates)
    {
        m_AnalogsDates.resize(analogsDates.rows(), analogsDates.cols());
        m_AnalogsDates = analogsDates;
    }

    /** Get the length of the target time dimension
     * \return The length of the target time
     */
    int GetTargetDatesLength()
    {
        return m_TargetDates.size();
    }

    /** Get the length of the analogs dimension
     * \return The length of the analogs
     */
    int GetAnalogsDatesLength()
    {
        return m_AnalogsDates.cols();
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
    Array1DFloat m_TargetDates; //!< Member variable "m_TargetDates"
    Array2DFloat m_AnalogsCriteria; //!< Member variable "m_AnalogCriteria"
    Array2DFloat m_AnalogsDates; //!< Member variable "m_AnalogDates"

};

#endif // ASRESULTSANALOGSDATES_H
