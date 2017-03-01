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

#ifndef ASRESULTSANALOGSVALUES_H
#define ASRESULTSANALOGSVALUES_H

#include <asIncludes.h>
#include <asResults.h>

class asResultsAnalogsValues
        : public asResults
{
public:
    asResultsAnalogsValues();

    virtual ~asResultsAnalogsValues();

    void Init(asParameters &params);

    Array1DFloat &GetTargetDates()
    {
        return m_targetDates;
    }

    void SetTargetDates(Array1DDouble &refDates)
    {
        m_targetDates.resize(refDates.rows());
        for (int i = 0; i < refDates.size(); i++) {
            m_targetDates[i] = (float) refDates[i];
            wxASSERT_MSG(m_targetDates[i] > 1, _("The target time array has unconsistent values"));
        }
    }

    void SetTargetDates(Array1DFloat &refDates)
    {
        m_targetDates.resize(refDates.rows());
        m_targetDates = refDates;
    }

    VArray1DFloat &GetTargetValues()
    {
        return m_targetValuesNorm;
    }

    void SetTargetValues(VArray1DFloat &targetValues)
    {
        m_targetValuesNorm = targetValues;
    }

    VArray1DFloat &GetTargetValuesNorm()
    {
        return m_targetValuesNorm;
    }

    void SetTargetValuesNorm(VArray1DFloat &targetValuesNorm)
    {
        m_targetValuesNorm = targetValuesNorm;
    }

    VArray1DFloat &GetTargetValuesGross()
    {
        return m_targetValuesGross;
    }

    void SetTargetValuesGross(VArray1DFloat &targetValuesGross)
    {
        m_targetValuesGross = targetValuesGross;
    }

    Array2DFloat &GetAnalogsCriteria()
    {
        return m_analogsCriteria;
    }

    void SetAnalogsCriteria(Array2DFloat &analogsCriteria)
    {
        m_analogsCriteria.resize(analogsCriteria.rows(), analogsCriteria.cols());
        m_analogsCriteria = analogsCriteria;
    }

    VArray2DFloat &GetAnalogsValues()
    {
        return m_analogsValuesNorm;
    }

    void SetAnalogsValues(VArray2DFloat &analogsValues)
    {
        m_analogsValuesNorm = analogsValues;
    }

    VArray2DFloat GetAnalogsValuesNorm() const
    {
        return m_analogsValuesNorm;
    }

    void SetAnalogsValuesNorm(VArray2DFloat &analogsValuesNorm)
    {
        m_analogsValuesNorm = analogsValuesNorm;
    }

    VArray2DFloat GetAnalogsValuesGross() const
    {
        return m_analogsValuesGross;
    }

    void SetAnalogsValuesGross(VArray2DFloat &analogsValuesGross)
    {
        m_analogsValuesGross = analogsValuesGross;
    }

    int GetTargetDatesLength() const
    {
        return m_targetDates.size();
    }

    bool Save();

    bool Load();

protected:
    void BuildFileName();

private:
    Array1DFloat m_targetDates; // Dimensions: time
    VArray1DFloat m_targetValuesNorm; // Dimensions: stations x time
    VArray1DFloat m_targetValuesGross; // Dimensions: stations x time
    Array2DFloat m_analogsCriteria; // Dimensions: time x analogs
    VArray2DFloat m_analogsValuesNorm; // Dimensions: stations x time x analogs
    VArray2DFloat m_analogsValuesGross; // Dimensions: stations x time x analogs
};

#endif // ASRESULTSANALOGSVALUES_H
