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
 * Portions Copyright 2014 Pascal Horton, Terr@num.
 */
 
#ifndef __asWizardBatchForecasts__
#define __asWizardBatchForecasts__

#include "AtmoswingForecasterGui.h"
#include "asBatchForecasts.h"

#include "asIncludes.h"

/** Implementing asWizardBatchForecasts */
class asWizardBatchForecasts : public asWizardBatchForecastsVirtual
{
public:
    /** Constructor */
    asWizardBatchForecasts( wxWindow* parent, asBatchForecasts* batchForecasts, wxWindowID id = wxID_ANY );
    ~asWizardBatchForecasts();
    
    wxWizardPage *GetFirstPage() const { return m_pages.Item( 0 ); }
    wxWizardPage *GetSecondPage() const { return m_pages.Item( 1 ); }

protected:
	void OnWizardFinished( wxWizardEvent& event );
	void OnLoadExistingBatchForecasts( wxCommandEvent& event );

private:
    asBatchForecasts* m_batchForecasts;
};

#endif // __asWizardBatchForecasts__
