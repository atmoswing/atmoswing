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
 
#ifndef ASDATAPREDICTORARCHIVE_H
#define ASDATAPREDICTORARCHIVE_H

#include <asIncludes.h>
#include <asDataPredictor.h>
#include <asCatalogPredictorsArchive.h>

class asGeoArea;

class asDataPredictorArchive: public asDataPredictor
{
public:

    /** Default constructor
     * \param catalog The predictor catalog
     */
    asDataPredictorArchive(asCatalogPredictorsArchive &catalog);

    /** Default destructor */
    virtual ~asDataPredictorArchive();

    bool LoadFullArea(double date, float level, const wxString &AlternatePredictorDataPath = wxEmptyString);

    virtual bool Load(asGeoAreaCompositeGrid &desiredArea, asTimeArray &timeArray, const VectorString &AlternatePredictorDataPath = VectorString(0));

    bool Load(asGeoAreaCompositeGrid &desiredArea, double date, const wxString &AlternatePredictorDataPath = wxEmptyString);
    bool Load(asGeoAreaCompositeGrid *desiredArea, double date, const wxString &AlternatePredictorDataPath = wxEmptyString);
    bool Load(asGeoAreaCompositeGrid &desiredArea, asTimeArray &timeArray, const wxString &AlternatePredictorDataPath);


    /** Method to load a tensor of data for a given area and a given time array
     * \param desiredArea The desired area
     * \param timeArray The desired time array
     */
    bool Load(asGeoAreaCompositeGrid *desiredArea, asTimeArray &timeArray, const wxString &AlternatePredictorDataPath = wxEmptyString);

    bool ClipToArea(asGeoAreaCompositeGrid *desiredArea);

    /** Access m_Catalog
     * \return The current value of m_Catalog
     */
    asCatalogPredictorsArchive& GetCatalog()
    {
        return m_Catalog;
    }


protected:
    asCatalogPredictorsArchive m_Catalog; //!< Member variable "m_Catalog"

    /** Method to check the time array compatibility with the catalog
     * \param timeArray The time array to check
     * \return True if compatible with the catalog
     */
    bool CheckTimeArray(asTimeArray &timeArray);

private:

};

#endif // ASDATAPREDICTORARCHIVE_H
