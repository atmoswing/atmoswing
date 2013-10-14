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
 
#ifndef ASDATAPREDICTORREALTIME_H
#define ASDATAPREDICTORREALTIME_H

#include <asIncludes.h>
#include <asDataPredictor.h>
#include <asCatalogPredictorsRealtime.h>


class asDataPredictorRealtime: public asDataPredictor
{
public:

    /** Default constructor
     * \param catalog The predictor catalog
     */
    asDataPredictorRealtime(asCatalogPredictorsRealtime &catalog);

    /** Default destructor */
    virtual ~asDataPredictorRealtime();

    int Download(asCatalogPredictorsRealtime &catalog);

    bool LoadFullArea(double date, float level, const VectorString &AlternatePredictorDataPath);

    virtual bool Load(asGeoAreaCompositeGrid &desiredArea, asTimeArray &timeArray, const VectorString &AlternatePredictorDataPath = VectorString(0));

    bool Load(asGeoAreaCompositeGrid *desiredArea, asTimeArray &timeArray, const VectorString &AlternatePredictorDataPath = VectorString(0));
    bool Load(asTimeArray &timeArray, const VectorString &AlternatePredictorDataPath = VectorString(0));

    /** Access m_Catalog
     * \return The current value of m_Catalog
     */
    asCatalogPredictorsRealtime& GetCatalog()
    {
        return m_Catalog;
    }


protected:
    asCatalogPredictorsRealtime m_Catalog; //!< Member variable "m_Catalog"

private:

};

#endif // ASDATAPREDICTORREALTIME_H
