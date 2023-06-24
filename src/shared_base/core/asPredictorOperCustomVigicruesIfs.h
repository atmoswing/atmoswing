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
 * Portions Copyright 2023 Pascal Horton, Terranum.
 */

#ifndef AS_PREDICTOR_OPER_VIGICRUES_IFS_H
#define AS_PREDICTOR_OPER_VIGICRUES_IFS_H

#include "asIncludes.h"
#include "asPredictorEcmwfIfs.h"
#include "asPredictorOperEcmwfIfs.h"

class asArea;

/**
 * @brief Operational predictor from the IFS model (ECMWF) with the Vigicrues naming convention.
 *
 * Operational predictor from the IFS model (ECMWF) with the Vigicrues naming convention.
 */
class asPredictorOperCustomVigicruesIfs : public asPredictorOperEcmwfIfs {
  public:
    /**
     * The constructor for the operational predictor class from IFS grib files with the Vigicrues naming convention.
     *
     * @param dataId Identifier of the data variable (meteorological parameter).
     */
    explicit asPredictorOperCustomVigicruesIfs(const wxString& dataId);

    /**
     * Destructor.
     */
    ~asPredictorOperCustomVigicruesIfs() override = default;

    /**
     * Get the file name from the data ID and the date.
     *
     * @param date Date of the model run for the file.
     * @param level Level of the data.
     * @return The file name.
     */
    wxString GetFileName(const double date, const int leadTime) override;

  protected:
  private:
};

#endif
