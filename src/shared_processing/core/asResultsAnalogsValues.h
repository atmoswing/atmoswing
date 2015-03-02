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

    /** Access m_targetValuesNorm
     * \return The whole array m_targetValuesNorm
     */
    VArray1DFloat &GetTargetValues()
    {
        return m_targetValuesNorm;
    }

    /** Set m_targetValuesNorm
     * \param targetValues The new array to set
     */
    void SetTargetValues(VArray1DFloat &targetValues)
    {
        m_targetValuesNorm = targetValues;
    }

    /** Access m_targetValuesNorm
     * \return The whole array m_targetValuesNorm
     */
    VArray1DFloat &GetTargetValuesNorm()
    {
        return m_targetValuesNorm;
    }

    /** Set m_targetValuesNorm
     * \param targetValuesNorm The new array to set
     */
    void SetTargetValuesNorm(VArray1DFloat &targetValuesNorm)
    {
        m_targetValuesNorm = targetValuesNorm;
    }

    /** Access m_targetValuesGross
     * \return The whole array m_targetValuesGross
     */
    VArray1DFloat &GetTargetValuesGross()
    {
        return m_targetValuesGross;
    }

    /** Set m_targetValuesGross
     * \param targetValuesGross The new array to set
     */
    void SetTargetValuesGross(VArray1DFloat &targetValuesGross)
    {
        m_targetValuesGross = targetValuesGross;
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

    /** Access m_analogValues
     * \return The whole array m_analogValues
     */
    VArray2DFloat &GetAnalogsValues()
    {
        return m_analogsValuesNorm;
    }

    /** Set m_analogValues
     * \param analogValues The new array to set
     */
    void SetAnalogsValues(VArray2DFloat &analogsValues)
    {
        m_analogsValuesNorm = analogsValues;
    }

    /** Access m_analogValuesNorm
     * \return The whole array m_analogValuesNorm
     */
    VArray2DFloat GetAnalogsValuesNorm()
    {
        return m_analogsValuesNorm;
    }

    /** Set m_analogValuesNorm
     * \param analogValuesNorm The new array to set
     */
    void SetAnalogsValuesNorm(VArray2DFloat &analogsValuesNorm)
    {
        m_analogsValuesNorm = analogsValuesNorm;
    }

    /** Access m_analogValuesGross
     * \return The whole array m_analogValuesGross
     */
    VArray2DFloat GetAnalogsValuesGross()
    {
        return m_analogsValuesGross;
    }

    /** Set m_analogValuesGross
     * \param analogValuesGross The new array to set
     */
    void SetAnalogsValuesGross(VArray2DFloat &analogsValuesGross)
    {
        m_analogsValuesGross = analogsValuesGross;
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
     */
    void BuildFileName();

private:
    Array1DFloat m_targetDates; //!< Member variable "m_targetDates". Dimensions: time
    VArray1DFloat m_targetValuesNorm; //!< Member variable "m_targetValuesNorm". Dimensions: stations x time
    VArray1DFloat m_targetValuesGross; //!< Member variable "m_targetValuesGross". Dimensions: stations x time
    Array2DFloat m_analogsCriteria; //!< Member variable "m_analogsCriteria". Dimensions: time x analogs
    VArray2DFloat m_analogsValuesNorm; //!< Member variable "m_analogsValuesNorm". Dimensions: stations x time x analogs
    VArray2DFloat m_analogsValuesGross; //!< Member variable "m_analogsValuesGross". Dimensions: stations x time x analogs
};

#endif // ASRESULTSANALOGSVALUES_H
