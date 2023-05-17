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

#ifndef AS_PREDICTOR_OPER_MF_ARPEGE_H
#define AS_PREDICTOR_OPER_MF_ARPEGE_H

#include "asIncludes.h"
#include "asPredictorOper.h"

class asArea;

class asPredictorOperMfArpege : public asPredictorOper {
  public:
    /**
     * The constructor for the operational ARPEGE predictor from Meteo France.
     *
     * @param dataId Identifier of the data variable (meteorological parameter).
     */
    explicit asPredictorOperMfArpege(const wxString& dataId);

    /**
     * Destructor.
     */
    ~asPredictorOperMfArpege() override = default;

    /**
     * Initialize the parameters of the data source.
     *
     * @return True if the initialisation went well.
     */
    bool Init() override;

    /**
     * Get the file name from the forecast date and the lead time.
     *
     * @param date The forecast date.
     * @param leadTime The lead time.
     * @return The file name.
     */
    wxString GetFileName(const double date, const int leadTime) override;

  protected:
    /**
     * Convert the tima array from hours to MJD.
     *
     * @param time The time array in hours (as in the files).
     * @param refValue The reference value to add to the time array (as in the files).
     */
    void ConvertToMjd(a1d& time, double refValue = NAN) const override;

  private:
};

#endif
