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

#ifndef ASRESULTSANALOGSDATES_H
#define ASRESULTSANALOGSDATES_H

#include <asIncludes.h>
#include <asResults.h>

class asResultsAnalogsDates
        : public asResults
{
public:
    asResultsAnalogsDates();

    virtual ~asResultsAnalogsDates();

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

    Array2DFloat &GetAnalogsCriteria()
    {
        return m_analogsCriteria;
    }

    void SetAnalogsCriteria(Array2DFloat &analogsCriteria)
    {
        m_analogsCriteria.resize(analogsCriteria.rows(), analogsCriteria.cols());
        m_analogsCriteria = analogsCriteria;
    }

    Array2DFloat &GetAnalogsDates()
    {
        return m_analogsDates;
    }

    void SetAnalogsDates(Array2DFloat &analogsDates)
    {
        m_analogsDates.resize(analogsDates.rows(), analogsDates.cols());
        m_analogsDates = analogsDates;
    }

    int GetTargetDatesLength() const
    {
        return (int) m_targetDates.size();
    }

    int GetAnalogsDatesLength() const
    {
        return (int) m_analogsDates.cols();
    }

    bool Save(const wxString &AlternateFilePath = wxEmptyString) const;

    bool Load(const wxString &AlternateFilePath = wxEmptyString);

protected:
    void BuildFileName();

private:
    Array1DFloat m_targetDates;
    Array2DFloat m_analogsCriteria;
    Array2DFloat m_analogsDates;

};

#endif // ASRESULTSANALOGSDATES_H
