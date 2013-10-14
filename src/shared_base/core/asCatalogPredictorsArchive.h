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
 
#ifndef ASCATALOGPREDICTORARCHIVES_H
#define ASCATALOGPREDICTORARCHIVES_H

#include <asIncludes.h>
#include <asCatalogPredictors.h>

class asCatalogPredictorsArchive: public asCatalogPredictors
{
public:

    /** Default constructor
     * \param DataSetId The dataset ID
     * \param DataId The data ID. If not set, load the whole database information
     */
    asCatalogPredictorsArchive(const wxString &alternateFilePath = wxEmptyString);

    /** Default destructor */
    virtual ~asCatalogPredictorsArchive();

    bool Load(const wxString &DataSetId, const wxString &DataId = wxEmptyString);

    /** Method that get dataset information from file
     * \param DataSetId The data Set ID
     */
    bool LoadDatasetProp(const wxString &DataSetId);

    /** Method that get data information from file
     * \param DataSetId The data Set ID
     * \param DataId The data ID
     */
    bool LoadDataProp(const wxString &DataSetId, const wxString &DataId);


protected:

private:

};

#endif // ASCATALOGPREDICTORARCHIVES_H
