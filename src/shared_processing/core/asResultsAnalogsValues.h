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
 
#ifndef ASRESULTSANALOGSVALUES_H
#define ASRESULTSANALOGSVALUES_H

#include <asIncludes.h>
#include <asResults.h>

class asResultsAnalogsValues: public asResults
{
public:

    /** Default constructor */
    asResultsAnalogsValues();

    /** Default destructor */
    virtual ~asResultsAnalogsValues();

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

    /** Access m_TargetValuesNorm
     * \return The whole array m_TargetValuesNorm
     */
    VArray1DFloat &GetTargetValues()
    {
        return m_TargetValuesNorm;
    }

    /** Set m_TargetValuesNorm
     * \param targetValues The new array to set
     */
    void SetTargetValues(VArray1DFloat &targetValues)
    {
        m_TargetValuesNorm = targetValues;
    }

    /** Access m_TargetValuesNorm
     * \return The whole array m_TargetValuesNorm
     */
    VArray1DFloat &GetTargetValuesNorm()
    {
        return m_TargetValuesNorm;
    }

    /** Set m_TargetValuesNorm
     * \param targetValuesNorm The new array to set
     */
    void SetTargetValuesNorm(VArray1DFloat &targetValuesNorm)
    {
        m_TargetValuesNorm = targetValuesNorm;
    }

    /** Access m_TargetValuesGross
     * \return The whole array m_TargetValuesGross
     */
    VArray1DFloat &GetTargetValuesGross()
    {
        return m_TargetValuesGross;
    }

    /** Set m_TargetValuesGross
     * \param targetValuesGross The new array to set
     */
    void SetTargetValuesGross(VArray1DFloat &targetValuesGross)
    {
        m_TargetValuesGross = targetValuesGross;
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

    /** Access m_AnalogValues
     * \return The whole array m_AnalogValues
     */
    VArray2DFloat &GetAnalogsValues()
    {
        return m_AnalogsValuesNorm;
    }

    /** Set m_AnalogValues
     * \param analogValues The new array to set
     */
    void SetAnalogsValues(VArray2DFloat &analogsValues)
    {
        m_AnalogsValuesNorm = analogsValues;
    }

    /** Access m_AnalogValuesNorm
     * \return The whole array m_AnalogValuesNorm
     */
    VArray2DFloat GetAnalogsValuesNorm()
    {
        return m_AnalogsValuesNorm;
    }

    /** Set m_AnalogValuesNorm
     * \param analogValuesNorm The new array to set
     */
    void SetAnalogsValuesNorm(VArray2DFloat &analogsValuesNorm)
    {
        m_AnalogsValuesNorm = analogsValuesNorm;
    }

    /** Access m_AnalogValuesGross
     * \return The whole array m_AnalogValuesGross
     */
    VArray2DFloat GetAnalogsValuesGross()
    {
        return m_AnalogsValuesGross;
    }

    /** Set m_AnalogValuesGross
     * \param analogValuesGross The new array to set
     */
    void SetAnalogsValuesGross(VArray2DFloat &analogsValuesGross)
    {
        m_AnalogsValuesGross = analogsValuesGross;
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
     */
    void BuildFileName();

private:
    Array1DFloat m_TargetDates; //!< Member variable "m_TargetDates". Dimensions: time
    VArray1DFloat m_TargetValuesNorm; //!< Member variable "m_TargetValuesNorm". Dimensions: stations x time
    VArray1DFloat m_TargetValuesGross; //!< Member variable "m_TargetValuesGross". Dimensions: stations x time
    Array2DFloat m_AnalogsCriteria; //!< Member variable "m_AnalogsCriteria". Dimensions: time x analogs
    VArray2DFloat m_AnalogsValuesNorm; //!< Member variable "m_AnalogsValuesNorm". Dimensions: stations x time x analogs
    VArray2DFloat m_AnalogsValuesGross; //!< Member variable "m_AnalogsValuesGross". Dimensions: stations x time x analogs
};

#endif // ASRESULTSANALOGSVALUES_H
