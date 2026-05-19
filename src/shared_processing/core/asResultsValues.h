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

#ifndef AS_RESULTS_VALUES_H
#define AS_RESULTS_VALUES_H

#include "asIncludes.h"
#include "asResults.h"

class asResultsValues : public asResults {
  public:
    asResultsValues();

    virtual ~asResultsValues();

    void Init(asParameters* arams);

    a1f& GetTargetDates() {
        return _targetDates;
    }

    void SetTargetDates(a1f& refDates) {
        _targetDates.resize(refDates.rows());
        _targetDates = refDates;
    }

    va1f& GetTargetValues() {
        return _targetValuesNorm;
    }

    void SetTargetValuesNorm(va1f& targetValuesNorm) {
        _targetValuesNorm = targetValuesNorm;
    }

    void SetTargetValuesRaw(va1f& targetValuesRaw) {
        _targetValuesRaw = targetValuesRaw;
    }

    a2f& GetAnalogsCriteria() {
        return _analogsCriteria;
    }

    void SetAnalogsCriteria(a2f& analogsCriteria) {
        _analogsCriteria.resize(analogsCriteria.rows(), analogsCriteria.cols());
        _analogsCriteria = analogsCriteria;
    }

    va2f& GetAnalogsValues() {
        return _analogsValuesNorm;
    }

    va2f& GetAnalogsValuesNorm() {
        return _analogsValuesNorm;
    }

    void SetAnalogsValuesNorm(va2f& analogsValuesNorm) {
        _analogsValuesNorm = analogsValuesNorm;
    }

    va2f GetAnalogsValuesRaw() const {
        return _analogsValuesRaw;
    }

    void SetAnalogsValuesRaw(va2f& analogsValuesRaw) {
        _analogsValuesRaw = analogsValuesRaw;
    }

    int GetTargetDatesLength() const {
        return _targetDates.size();
    }

    bool Save();

    bool Load();

  protected:
    void BuildFileName();

  private:
    a1f _targetDates;         // Dimensions: time
    va1f _targetValuesNorm;   // Dimensions: stations x time
    va1f _targetValuesRaw;    // Dimensions: stations x time
    a2f _analogsCriteria;     // Dimensions: time x analogs
    va2f _analogsValuesNorm;  // Dimensions: stations x time x analogs
    va2f _analogsValuesRaw;   // Dimensions: stations x time x analogs
};

#endif
