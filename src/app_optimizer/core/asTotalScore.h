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

#ifndef AS_TOTAL_SCORE_H
#define AS_TOTAL_SCORE_H

#include "asIncludes.h"
#include "asTimeArray.h"

class asTotalScore : public wxObject {
  public:
    enum Period  //!< Enumaration of forcast score combinations
    {
        Total,           // total mean
        SpecificPeriod,  // partial mean
        Summer,          // partial mean on summer only
        Fall,            // partial mean on fall only
        Winter,          // partial mean on winter only
        Spring,          // partial mean on spring only
    };

    explicit asTotalScore(const wxString& periodString);

    ~asTotalScore() override;

    static asTotalScore* GetInstance(const wxString& scoreString, const wxString& periodString);

    virtual float Assess(const a1f& targetDates, const a1f& scores, const asTimeArray& timeArray) const = 0;

    virtual float Assess(const a1f& targetDates, const a2f& scores, const asTimeArray& timeArray) const;

    virtual a1f AssessOnArray(const a1f& targetDates, const a1f& scores, const asTimeArray& timeArray) const;

    bool SingleValue() const {
        return m_singleValue;
    }

    bool Has2DArrayArgument() const {
        return m_has2DArrayArgument;
    }

    void SetRanksNb(int val) {
        m_ranksNb = val;
    }

  protected:
    Period m_period;
    bool m_singleValue;
    bool m_has2DArrayArgument;
    int m_ranksNb;

  private:
};

#endif
