"""Module for translating output from fusion detection methods to fusion
objects (AssayedFusion/CategoricalFusion)
"""

import logging
import re
from abc import ABC, abstractmethod
from enum import Enum

import polars as pl
from civicpy.civic import ExonCoordinate, MolecularProfile
from cool_seq_tool.schemas import Assembly, CoordinateType
from ga4gh.core.models import MappableConcept
from ga4gh.vrs.models import LiteralSequenceExpression
from pydantic import BaseModel

from fusor.fusion_caller_models import (
    CIVIC,
    JAFFA,
    Arriba,
    Caller,
    Cicero,
    EnFusion,
    FusionCatcher,
    Genie,
    KnowledgebaseList,
    STARFusion,
)
from fusor.fusor import FUSOR
from fusor.models import (
    LINKER_REGEX,
    AnchoredReads,
    Assay,
    AssayedFusion,
    BreakpointCoverage,
    CategoricalFusion,
    CausativeEvent,
    ContigSequence,
    EventType,
    GeneElement,
    LinkerElement,
    MultiplePossibleGenesElement,
    ReadData,
    SpanningReads,
    SplitReads,
    TranscriptSegmentElement,
    UnknownGeneElement,
)

_logger = logging.getLogger(__name__)


class GeneFusionPartners(BaseModel):
    """Class for defining gene fusion partners"""

    gene_5prime_element: GeneElement | UnknownGeneElement | MultiplePossibleGenesElement
    gene_5prime: str | None = None
    gene_3prime_element: GeneElement | UnknownGeneElement | MultiplePossibleGenesElement
    gene_3prime: str | None = None


class Translator(ABC):
    """Class for translating outputs from different fusion detection algorithms
    to FUSOR AssayedFusion and CategoricalFusion objects
    """

    def __init__(self, fusor: FUSOR) -> None:
        """Initialize Translator class

        :param fusor: A FUSOR instance
        """
        self.fusor = fusor

    def _format_fusion(
        self,
        fusion_type: AssayedFusion | CategoricalFusion,
        gene_5prime: GeneElement | UnknownGeneElement | MultiplePossibleGenesElement,
        gene_3prime: GeneElement | UnknownGeneElement | MultiplePossibleGenesElement,
        tr_5prime: TranscriptSegmentElement | None = None,
        tr_3prime: TranscriptSegmentElement | None = None,
        ce: CausativeEvent | None = None,
        rf: bool | None = None,
        assay: Assay | None = None,
        contig: ContigSequence | None = None,
        linker_sequence: LinkerElement | None = None,
        reads: ReadData | None = None,
        molecular_profiles: list[MolecularProfile] | None = None,
        moa_assertion: dict | None = None,
    ) -> AssayedFusion | CategoricalFusion:
        """Format classes to create AssayedFusion objects

        :param fusion_type: If the fusion is an AssayedFusion or CategoricalFusion
        :param gene_5prime: 5'prime GeneElement or UnknownGeneElement or MultiplePossibleGenesElement
        :param gene_3prime: 3'prime GeneElement or UnknownGeneElement or MultiplePossibleGenesElement
        :param tr_5prime: 5'prime TranscriptSegmentElement
        :param tr_3prime: 3'prime TranscriptSegmentElement
        :param ce: CausativeEvent
        :param rf: A boolean indicating if the reading frame is preserved
        :param assay: Assay
        :param contig: The contig sequence
        :param linker_sequence: The non-template linker sequence
        :param reads: The read data
        :param molecular_profiles: A list of CIViC Molecular Profiles
        :param moa_assertion: The MOA assertion, represented as a dictionary
        :return AssayedFusion or CategoricalFusion object
        """
        params = {
            "causativeEvent": ce,
            "readingFramePreserved": rf,
            "assay": assay,
            "contig": contig,
            "readData": reads,
            "civicMolecularProfiles": molecular_profiles,
            "moaAssertion": moa_assertion,
        }
        if not tr_5prime and not tr_3prime:
            params["structure"] = [gene_5prime, gene_3prime]
        elif tr_5prime and not tr_3prime:
            params["structure"] = [tr_5prime, gene_3prime]
        elif not tr_5prime and tr_3prime:
            params["structure"] = [gene_5prime, tr_3prime]
        else:
            params["structure"] = [tr_5prime, tr_3prime]
        if linker_sequence:
            params["structure"].insert(1, linker_sequence)
        fusion = fusion_type(**params)

        # Assign VICC Nomenclature string to fusion event
        fusion.viccNomenclature = self.fusor.generate_nomenclature(fusion)
        return fusion

    def _get_causative_event(
        self, chrom1: str, chrom2: str, descr: str | None = None
    ) -> CausativeEvent | None:
        """Infer Causative Event. Currently restricted to rearrangements

        :param chrom1: The chromosome for the 5' partner
        :param chrom2: The chromosome for the 3' partner
        :param descr: An annotation describing the fusion event. This input is supplied to the eventDescription CausativeEvent attribute.
        :return: A CausativeEvent object if construction is successful
        """
        if descr and "rearrangement" in descr:
            return CausativeEvent(
                eventType=EventType("rearrangement"), eventDescription=descr
            )
        if chrom1 != chrom2:
            return CausativeEvent(eventType=EventType("rearrangement"))
        return None

    def _get_gene_element_unnormalized(self, symbol: str) -> GeneElement:
        """Return GeneElement when gene symbol cannot be normalized

        :param symbol: A gene symbol for a fusion partner
        :return: A GeneElement object
        """
        return GeneElement(gene=MappableConcept(name=symbol, conceptType="Gene"))

    def _get_gene_element(
        self, genes: str, caller: Caller | KnowledgebaseList
    ) -> GeneElement:
        """Return a GeneElement given an individual/list of gene symbols and a
        fusion detection algorithm

        :param genes: A gene symbol or list of gene symbols, separated by columns
        :param caller: The examined fusion detection algorithm of fusion knowledgebase
        :return A GeneElement object
        """
        if "," not in genes or caller != caller.ARRIBA:
            ge = self.fusor.gene_element(gene=genes)
            return ge[0] if ge[0] else self._get_gene_element_unnormalized(genes)

        genes = genes.split(",")
        dists = []
        for gene in genes:
            start, end = gene.rfind("("), gene.rfind(")")
            dists.append(int(gene[start + 1 : end]))
        gene = (
            genes[0].split("(")[0] if dists[0] <= dists[1] else genes[1].split("(")[0]
        )
        ge = self.fusor.gene_element(gene=gene)
        return ge[0] if ge[0] else self._get_gene_element_unnormalized(gene)

    def _are_fusion_partners_different(
        self,
        gene_5prime: str | UnknownGeneElement | MultiplePossibleGenesElement,
        gene_3prime: str | UnknownGeneElement | MultiplePossibleGenesElement,
    ) -> bool:
        """Check if the normalized gene symbols for the two fusion partners
        are different. If not, this event is not a fusion

        :param gene_5prime: The 5' gene partner or UnknownGeneElement or MultiplePossibleGenesElement
        :param gene_3prime: The 3' gene partner or UnknownGeneElement or MultiplePossibleGenesElement
        :return ``True`` if the gene symbols are different, ``False`` if not
        """
        if gene_5prime != gene_3prime:
            return True
        _logger.error(
            "The supplied fusion is not valid as the two fusion partners are the same"
        )
        return False

    def _get_genomic_ac(self, chrom: str, build: Assembly | str) -> str:
        """Return a RefSeq genomic accession given a chromosome and a reference build

        :param chrom: A chromosome number
        :param build: The assembly, either GRCh37 or GRCh38
        :return: The corresponding refseq genomic accession
        :raise ValueError: if unable to retrieve genomic accession
        """
        sr = self.fusor.cool_seq_tool.seqrepo_access
        build_id = build.value if isinstance(build, Assembly) else build
        alias_list, errors = sr.translate_identifier(
            f"{build_id}:{chrom}", target_namespaces="refseq"
        )
        if errors:
            statement = f"Genomic accession for {chrom} could not be retrieved"
            _logger.error(statement)
            raise ValueError
        return alias_list[0].split(":")[1]

    def _assess_gene_symbol(
        self, gene: str | None, caller: Caller | KnowledgebaseList
    ) -> (
        tuple[GeneElement | UnknownGeneElement | MultiplePossibleGenesElement, str]
        | None
    ):
        """Determine if a gene symbol exists and return the corresponding
        GeneElement

        :param gene: The gene symbol or None
        :param caller: The gene fusion caller or fusion knowledgebase
        :return A tuple containing a GeneElement or UnknownGeneElement and a string,
            representing the unknown fusion partner, or MultiplePossibleGenesElement
            and a string, representing any possible fusion partner or None if no gene
            is provided
        """
        if not gene:
            return None
        if gene == "NA":
            return UnknownGeneElement(), "NA"
        if gene == "v":
            return MultiplePossibleGenesElement(), "v"
        gene_element = self._get_gene_element(gene, caller)
        return gene_element, gene_element.gene.name

    def _process_gene_symbols(
        self, gene_5prime: str, gene_3prime: str, caller: Caller | KnowledgebaseList
    ) -> GeneFusionPartners:
        """Process gene symbols to create GeneElements or UnknownGeneElements or MultiplePossibleGenesElement

        :param gene_5prime: The 5' gene symbol
        :param gene_3prime: The 3' gene symbol
        :param caller: The gene fusion caller or fusion knowledgebase
        :return A GeneFusionPartners object
        """
        gene_5prime_element, gene_5prime = self._assess_gene_symbol(gene_5prime, caller)
        gene_3prime_element, gene_3prime = self._assess_gene_symbol(gene_3prime, caller)
        params = {
            "gene_5prime_element": gene_5prime_element,
            "gene_5prime": gene_5prime,
            "gene_3prime_element": gene_3prime_element,
            "gene_3prime": gene_3prime,
        }
        return GeneFusionPartners(**params)

    def _process_vicc_nomenclature(self, gene_symbol: str) -> str:
        """Extract fusion partner from VICC nomenclature

        :param gene_symbol: The unprocessed gene symbol
        :return The processed gene symbol
        """
        if "entrez" in gene_symbol:
            return gene_symbol.split("(")[0]
        start = gene_symbol.find("(")
        stop = gene_symbol.find(")")
        return gene_symbol[start + 1 : stop]

    @abstractmethod
    async def translate(
        self, fusion_data: BaseModel, coordinate_type: CoordinateType, rb: Assembly
    ) -> AssayedFusion | CategoricalFusion:
        """Define abstract translate method

        :param fusion_data: The fusion data from a fusion caller
        :param coordinate_type: If the coordinate is inter-residue or residue
        :param rb: The reference build used to call the fusion
        :return: An AssayedFusion or CategoricalFusion object, if construction is successful
        """

    ##### Fusion Caller -> FUSOR AssayedFusion object ###################


class JAFFATranslator(Translator):
    """Initialize JAFFATranslator class"""

    async def translate(
        self,
        jaffa: JAFFA,
        coordinate_type: CoordinateType,
        rb: Assembly,
    ) -> AssayedFusion | None:
        """Parse JAFFA fusion output to create AssayedFusion object

        :param JAFFA: Output from JAFFA caller
        :param coordinate_type: If the coordinate is inter-residue or residue
        :param rb: The reference build used to call the fusion
        :return: An AssayedFusion object, if construction is successful
        """
        genes = jaffa.fusion_genes.split(":")
        fusion_partners = self._process_gene_symbols(genes[0], genes[1], Caller.JAFFA)

        if not self._are_fusion_partners_different(
            fusion_partners.gene_5prime, fusion_partners.gene_3prime
        ):
            return None

        if not isinstance(fusion_partners.gene_5prime_element, UnknownGeneElement):
            tr_5prime = await self.fusor.transcript_segment_element(
                tx_to_genomic_coords=False,
                genomic_ac=self._get_genomic_ac(jaffa.chrom1, rb),
                seg_end_genomic=jaffa.base1,
                gene=fusion_partners.gene_5prime,
                coordinate_type=coordinate_type,
                starting_assembly=rb,
            )
            tr_5prime = tr_5prime[0]

        if not isinstance(fusion_partners.gene_3prime_element, UnknownGeneElement):
            tr_3prime = await self.fusor.transcript_segment_element(
                tx_to_genomic_coords=False,
                genomic_ac=self._get_genomic_ac(jaffa.chrom2, rb),
                seg_start_genomic=jaffa.base2,
                gene=fusion_partners.gene_3prime,
                coordinate_type=coordinate_type,
                starting_assembly=rb,
            )
            tr_3prime = tr_3prime[0]

        if jaffa.rearrangement:
            ce = CausativeEvent(
                eventType=EventType("rearrangement"),
                eventDescription=jaffa.classification,
            )
        else:
            ce = None

        read_data = ReadData(
            split=SplitReads(splitReads=jaffa.spanning_reads),
            spanning=SpanningReads(spanningReads=jaffa.spanning_pairs),
        )

        return self._format_fusion(
            AssayedFusion,
            fusion_partners.gene_5prime_element,
            fusion_partners.gene_3prime_element,
            tr_5prime
            if isinstance(fusion_partners.gene_5prime_element, GeneElement)
            else None,
            tr_3prime
            if isinstance(fusion_partners.gene_3prime_element, GeneElement)
            else None,
            ce,
            jaffa.inframe if isinstance(jaffa.inframe, bool) else None,
            reads=read_data,
        )


class STARFusionTranslator(Translator):
    """Initialize STARFusionTranslator class"""

    async def translate(
        self,
        star_fusion: STARFusion,
        coordinate_type: CoordinateType,
        rb: Assembly,
    ) -> AssayedFusion:
        """Parse STAR-Fusion output to create AssayedFusion object

        :param star_fusion: Output from STAR-Fusion caller
        :param coordinate_type: If the coordinate is inter-residue or residue
        :param rb: The reference build used to call the fusion
        :return: An AssayedFusion object, if construction is successful
        """
        gene1 = star_fusion.left_gene.split("^")[0]
        gene2 = star_fusion.right_gene.split("^")[0]

        fusion_partners = self._process_gene_symbols(gene1, gene2, Caller.STAR_FUSION)
        if not self._are_fusion_partners_different(
            fusion_partners.gene_5prime, fusion_partners.gene_3prime
        ):
            return None

        five_prime = star_fusion.left_breakpoint.split(":")
        three_prime = star_fusion.right_breakpoint.split(":")

        tr_5prime = await self.fusor.transcript_segment_element(
            tx_to_genomic_coords=False,
            genomic_ac=self._get_genomic_ac(five_prime[0], rb),
            seg_end_genomic=int(five_prime[1]),
            gene=fusion_partners.gene_5prime,
            coordinate_type=coordinate_type,
            starting_assembly=rb,
        )
        tr_5prime = tr_5prime[0]

        tr_3prime = await self.fusor.transcript_segment_element(
            tx_to_genomic_coords=False,
            genomic_ac=self._get_genomic_ac(three_prime[0], rb),
            seg_start_genomic=int(three_prime[1]),
            gene=fusion_partners.gene_3prime,
            coordinate_type=coordinate_type,
            starting_assembly=rb,
        )
        tr_3prime = tr_3prime[0]

        ce = self._get_causative_event(
            five_prime[0], three_prime[0], ",".join(star_fusion.annots)
        )
        read_data = ReadData(
            split=SplitReads(splitReads=star_fusion.junction_read_count),
            spanning=SpanningReads(spanningReads=star_fusion.spanning_frag_count),
        )

        return self._format_fusion(
            AssayedFusion,
            fusion_partners.gene_5prime_element,
            fusion_partners.gene_3prime_element,
            tr_5prime
            if isinstance(fusion_partners.gene_5prime_element, GeneElement)
            else None,
            tr_3prime
            if isinstance(fusion_partners.gene_3prime_element, GeneElement)
            else None,
            ce,
            reads=read_data,
        )


class FusionCatcherTranslator(Translator):
    """Initialize FusionCatcherTranslator class"""

    async def translate(
        self,
        fusion_catcher: FusionCatcher,
        coordinate_type: CoordinateType,
        rb: Assembly,
    ) -> AssayedFusion:
        """Parse FusionCatcher output to create AssayedFusion object

        :param fusion_catcher: Output from FusionCatcher caller
        :param coordinate_type: If the coordinate is inter-residue or residue
        :param rb: The reference build used to call the fusion
        :return: An AssayedFusion object, if construction is successful
        """
        fusion_partners = self._process_gene_symbols(
            fusion_catcher.five_prime_partner,
            fusion_catcher.three_prime_partner,
            Caller.FUSION_CATCHER,
        )
        if not self._are_fusion_partners_different(
            fusion_partners.gene_5prime, fusion_partners.gene_3prime
        ):
            return None

        five_prime = fusion_catcher.five_prime_fusion_point.split(":")
        three_prime = fusion_catcher.three_prime_fusion_point.split(":")

        tr_5prime = await self.fusor.transcript_segment_element(
            tx_to_genomic_coords=False,
            genomic_ac=self._get_genomic_ac(five_prime[0], rb),
            seg_end_genomic=int(five_prime[1]),
            gene=fusion_partners.gene_5prime,
            coordinate_type=coordinate_type,
            starting_assembly=rb,
        )
        tr_5prime = tr_5prime[0]

        tr_3prime = await self.fusor.transcript_segment_element(
            tx_to_genomic_coords=False,
            genomic_ac=self._get_genomic_ac(three_prime[0], rb),
            seg_start_genomic=int(three_prime[1]),
            gene=fusion_partners.gene_3prime,
            coordinate_type=coordinate_type,
            starting_assembly=rb,
        )
        tr_3prime = tr_3prime[0]

        ce = self._get_causative_event(
            five_prime[0], three_prime[0], fusion_catcher.predicted_effect
        )
        read_data = ReadData(
            split=SplitReads(splitReads=fusion_catcher.spanning_unique_reads),
            spanning=SpanningReads(spanningReads=fusion_catcher.spanning_reads),
        )
        contig = ContigSequence(contig=fusion_catcher.fusion_sequence)

        return self._format_fusion(
            AssayedFusion,
            fusion_partners.gene_5prime_element,
            fusion_partners.gene_3prime_element,
            tr_5prime
            if isinstance(fusion_partners.gene_5prime_element, GeneElement)
            else None,
            tr_3prime
            if isinstance(fusion_partners.gene_3prime_element, GeneElement)
            else None,
            ce,
            contig=contig,
            reads=read_data,
        )


class FusionMapTranslator(Translator):
    """Initialize FusionMapTranslator class"""

    async def translate(
        self, fmap_row: pl.DataFrame, coordinate_type: CoordinateType, rb: Assembly
    ) -> AssayedFusion:
        """Parse FusionMap output to create FUSOR AssayedFusion object

        :param fmap_row: A row of FusionMap output
        :param rb: The reference build used to call the fusion
        :param coordinate_type: If the coordinate is inter-residue or residue
        :return: An AssayedFusion object, if construction is successful
        """
        gene1 = fmap_row.get_column("KnownGene1").item()
        gene2 = fmap_row.get_column("KnownGene2").item()
        gene_5prime = self._get_gene_element(gene1, "fusion_map").gene.name
        gene_3prime = self._get_gene_element(gene2, "fusion_map").gene.name

        if not self._are_fusion_partners_different(gene_5prime, gene_3prime):
            return None

        tr_5prime = await self.fusor.transcript_segment_element(
            tx_to_genomic_coords=False,
            genomic_ac=self._get_genomic_ac(
                fmap_row.get_column("Chromosome1").item(), rb
            ),
            seg_end_genomic=int(fmap_row.get_column("Position1").item()),
            gene=gene_5prime,
            coordinate_type=coordinate_type,
            starting_assembly=rb,
        )
        tr_5prime = tr_5prime[0]

        tr_3prime = await self.fusor.transcript_segment_element(
            tx_to_genomic_coords=False,
            genomic_ac=self._get_genomic_ac(
                fmap_row.get_column("Chromosome2").item(), rb
            ),
            seg_start_genomic=int(fmap_row.get_column("Position2").item()),
            gene=gene_3prime,
            coordinate_type=coordinate_type,
            starting_assembly=rb,
        )
        tr_3prime = tr_3prime[0]

        # Combine columns to create fusion annotation string"
        descr = (
            fmap_row.get_column("FusionGene").item()
            + ","
            + fmap_row.get_column("SplicePatternClass").item()
            + ","
            + fmap_row.get_column("FrameShiftClass").item()
        )
        ce = self._get_causative_event(
            fmap_row.get_column("Chromosome1").item(),
            fmap_row.get_column("Chromosome2").item(),
            descr,
        )
        rf = bool(fmap_row.get_column("FrameShiftClass").item() == "InFrame")
        return self._format_fusion(
            AssayedFusion, gene_5prime, gene_3prime, tr_5prime, tr_3prime, ce, rf
        )


class ArribaTranslator(Translator):
    """Initialize ArribaTranslator class"""

    async def translate(
        self,
        arriba: Arriba,
        coordinate_type: CoordinateType,
        rb: Assembly,
    ) -> AssayedFusion:
        """Parse Arriba output to create AssayedFusion object

        :param arriba: Output from Arriba caller
        :param coordinate_type: If the coordinate is inter-residue or residue
        :param rb: The reference build used to call the fusion
        :return: An AssayedFusion object, if construction is successful
        """
        # Arriba reports two gene symbols if a breakpoint occurs in an intergenic
        # space. We select the gene symbol with the smallest distance from the
        # breakpoint.
        fusion_partners = self._process_gene_symbols(
            arriba.gene1, arriba.gene2, Caller.ARRIBA
        )

        if not self._are_fusion_partners_different(
            fusion_partners.gene_5prime, fusion_partners.gene_3prime
        ):
            return None

        strand1 = arriba.strand1.split("/")[1]  # Determine strand that is transcribed
        strand2 = arriba.strand2.split("/")[1]  # Determine strand that is transcribed
        if strand1 == "+":
            gene1_seg_start = arriba.direction1 == "upstream"
        else:
            gene1_seg_start = arriba.direction1 == "downstream"
        if strand2 == "+":
            gene2_seg_start = arriba.direction2 == "upstream"
        else:
            gene2_seg_start = arriba.direction2 == "downstream"

        breakpoint1 = arriba.breakpoint1.split(":")
        breakpoint2 = arriba.breakpoint2.split(":")

        tr_5prime = await self.fusor.transcript_segment_element(
            tx_to_genomic_coords=False,
            genomic_ac=self._get_genomic_ac(breakpoint1[0], rb),
            seg_start_genomic=int(breakpoint1[1]) if gene1_seg_start else None,
            seg_end_genomic=int(breakpoint1[1]) if not gene1_seg_start else None,
            gene=fusion_partners.gene_5prime,
            coverage=BreakpointCoverage(fragmentCoverage=arriba.coverage1),
            reads=AnchoredReads(reads=arriba.split_reads1),
            coordinate_type=coordinate_type,
            starting_assembly=rb,
        )
        tr_5prime = tr_5prime[0]

        tr_3prime = await self.fusor.transcript_segment_element(
            tx_to_genomic_coords=False,
            genomic_ac=self._get_genomic_ac(breakpoint2[0], rb),
            seg_start_genomic=int(breakpoint2[1]) if gene2_seg_start else None,
            seg_end_genomic=int(breakpoint2[1]) if not gene2_seg_start else None,
            gene=fusion_partners.gene_3prime,
            coverage=BreakpointCoverage(fragmentCoverage=arriba.coverage2),
            reads=AnchoredReads(reads=arriba.split_reads2),
            coordinate_type=coordinate_type,
            starting_assembly=rb,
        )
        tr_3prime = tr_3prime[0]

        ce = (
            CausativeEvent(
                eventType=EventType("read-through"),
                eventDescription=arriba.confidence,
            )
            if "read_through" in arriba.event_type
            else CausativeEvent(
                eventType=EventType("rearrangement"),
                eventDescription=arriba.confidence,
            )
        )
        rf = bool(arriba.rf == "in-frame") if arriba.rf != "." else None

        # Process read data and fusion_transcript sequence
        read_data = ReadData(
            spanning=SpanningReads(spanningReads=arriba.discordant_mates)
        )
        contig = ContigSequence(contig=arriba.fusion_transcript)
        linker_sequence = re.search(LINKER_REGEX, arriba.fusion_transcript)
        linker_sequence = (
            LinkerElement(
                linkerSequence=LiteralSequenceExpression(
                    sequence=linker_sequence.group(1).upper()
                )
            )
            if linker_sequence
            else None
        )

        return self._format_fusion(
            AssayedFusion,
            fusion_partners.gene_5prime_element,
            fusion_partners.gene_3prime_element,
            tr_5prime
            if isinstance(fusion_partners.gene_5prime_element, GeneElement)
            else None,
            tr_3prime
            if isinstance(fusion_partners.gene_3prime_element, GeneElement)
            else None,
            ce,
            rf,
            contig=contig,
            reads=read_data,
            linker_sequence=linker_sequence,
        )


class CiceroTranslator(Translator):
    """Initialize CiceroTranslator class"""

    async def translate(
        self,
        cicero: Cicero,
        coordinate_type: CoordinateType,
        rb: Assembly,
    ) -> AssayedFusion | str:
        """Parse CICERO output to create AssayedFusion object

        :param cicero: Output from CICERO caller
        :param coordinate_type: If the coordinate is inter-residue or residue
        :param rb: The reference build used to call the fusion
        :return: An AssayedFusion object, if construction is successful
        """
        # Check if gene symbols have valid formatting. CICERO can output two or more
        # gene symbols for `gene_5prime` or `gene_3prime`, which are separated by a comma. As
        # there is not a precise way to resolve this ambiguity, we do not process
        # these events
        if "," in cicero.gene_5prime or "," in cicero.gene_3prime:
            msg = "Ambiguous gene symbols are reported by CICERO for at least one of the fusion partners"
            _logger.warning(msg)
            return msg

        fusion_partners = self._process_gene_symbols(
            cicero.gene_5prime, cicero.gene_3prime, Caller.CICERO
        )

        # Check CICERO annotation regarding the confidence that the called fusion
        # has biological meaning
        if cicero.sv_ort != ">":
            msg = "CICERO annotation indicates that this event does not have confident biological meaning"
            _logger.warning(msg)
            return msg

        if not self._are_fusion_partners_different(
            fusion_partners.gene_5prime, fusion_partners.gene_3prime
        ):
            return None

        tr_5prime = await self.fusor.transcript_segment_element(
            tx_to_genomic_coords=False,
            genomic_ac=self._get_genomic_ac(cicero.chr_5prime, rb),
            seg_end_genomic=cicero.pos_5prime,
            gene=fusion_partners.gene_5prime,
            coverage=BreakpointCoverage(fragmentCoverage=cicero.coverage_5prime),
            reads=AnchoredReads(reads=cicero.reads_5prime),
            coordinate_type=coordinate_type,
            starting_assembly=rb,
        )
        tr_5prime = tr_5prime[0]

        tr_3prime = await self.fusor.transcript_segment_element(
            tx_to_genomic_coords=False,
            genomic_ac=self._get_genomic_ac(cicero.chr_3prime, rb),
            seg_start_genomic=cicero.pos_3prime,
            gene=fusion_partners.gene_3prime,
            coverage=BreakpointCoverage(fragmentCoverage=cicero.coverage_3prime),
            reads=AnchoredReads(reads=cicero.reads_3prime),
            coordinate_type=coordinate_type,
            starting_assembly=rb,
        )
        tr_3prime = tr_3prime[0]

        if cicero.event_type == "read_through":
            ce = CausativeEvent(
                eventType=EventType("read-through"),
                eventDescription=cicero.event_type,
            )
        else:
            ce = CausativeEvent(
                eventType=EventType("rearrangement"),
                eventDescription=cicero.event_type,
            )
        contig = ContigSequence(contig=cicero.contig)

        return self._format_fusion(
            AssayedFusion,
            fusion_partners.gene_5prime_element,
            fusion_partners.gene_3prime_element,
            tr_5prime
            if isinstance(fusion_partners.gene_5prime_element, GeneElement)
            else None,
            tr_3prime
            if isinstance(fusion_partners.gene_3prime_element, GeneElement)
            else None,
            ce,
            contig=contig,
        )


class MapSpliceTranslator(Translator):
    """Initialize MapSpliceTranslator class"""

    async def translate(
        self, mapsplice_row: pl.DataFrame, coordinate_type: CoordinateType, rb: Assembly
    ) -> AssayedFusion:
        """Parse MapSplice output to create AssayedFusion object

        :param mapsplice_row: A row of MapSplice output
        :param rb: The reference build used to call the fusion
        :param coordinate_type: If the coordinate is inter-residue or residue
        :retun: An AssayedFusion object, if construction is successful
        """
        gene1 = mapsplice_row[60].strip(",")
        gene2 = mapsplice_row[61].strip(",")
        gene_5prime_element = self._get_gene_element(gene1, "mapsplice")
        gene_3prime_element = self._get_gene_element(gene2, "mapsplice")
        gene_5prime = gene_5prime_element.gene.name
        gene_3prime = gene_3prime_element.gene.name

        if not self._are_fusion_partners_different(gene_5prime, gene_3prime):
            return None

        tr_5prime = await self.fusor.transcript_segment_element(
            tx_to_genomic_coords=False,
            genomic_ac=self._get_genomic_ac(mapsplice_row[0].split("~")[0], rb),
            seg_end_genomic=int(mapsplice_row[1]),
            gene=gene_5prime,
            coordinate_type=coordinate_type,
            starting_assembly=rb,
        )
        tr_5prime = tr_5prime[0]

        tr_3prime = await self.fusor.transcript_segment_element(
            tx_to_genomic_coords=False,
            genomic_ac=self._get_genomic_ac(mapsplice_row[0].split("~")[1], rb),
            seg_start_genomic=int(mapsplice_row[2]),
            gene=gene_3prime,
            coordinate_type=coordinate_type,
            starting_assembly=rb,
        )
        tr_3prime = tr_3prime[0]

        ce = self._get_causative_event(
            mapsplice_row[0].split("~")[0], mapsplice_row[0].split("~")[1]
        )
        return self._format_fusion(
            AssayedFusion, gene_5prime, gene_3prime, tr_5prime, tr_3prime, ce
        )


class EnFusionTranslator(Translator):
    """Initialize EnFusionTranslator class"""

    async def translate(
        self,
        enfusion: EnFusion,
        coordinate_type: CoordinateType,
        rb: Assembly,
    ) -> AssayedFusion:
        """Parse EnFusion output to create AssayedFusion object

        :param enfusion: Output from EnFusion caller
        :param coordinate_type: If the coordinate is inter-residue or residue
        :param rb: The reference build used to call the fusion
        :return: An AssayedFusion object, if construction is successful
        """
        fusion_partners = self._process_gene_symbols(
            enfusion.gene_5prime, enfusion.gene_3prime, Caller.ENFUSION
        )

        if not self._are_fusion_partners_different(
            fusion_partners.gene_5prime, fusion_partners.gene_3prime
        ):
            return None

        tr_5prime = await self.fusor.transcript_segment_element(
            tx_to_genomic_coords=False,
            genomic_ac=self._get_genomic_ac(enfusion.chr_5prime, rb),
            seg_end_genomic=enfusion.break_5prime,
            gene=fusion_partners.gene_5prime,
            coordinate_type=coordinate_type,
            starting_assembly=rb,
        )
        tr_5prime = tr_5prime[0]

        tr_3prime = await self.fusor.transcript_segment_element(
            tx_to_genomic_coords=False,
            genomic_ac=self._get_genomic_ac(enfusion.chr_3prime, rb),
            seg_start_genomic=enfusion.break_3prime,
            gene=fusion_partners.gene_3prime,
            coordinate_type=coordinate_type,
            starting_assembly=rb,
        )
        tr_3prime = tr_3prime[0]

        ce = self._get_causative_event(
            enfusion.chr_5prime,
            enfusion.chr_3prime,
        )
        return self._format_fusion(
            AssayedFusion,
            fusion_partners.gene_5prime_element,
            fusion_partners.gene_3prime_element,
            tr_5prime
            if isinstance(fusion_partners.gene_5prime_element, GeneElement)
            else None,
            tr_3prime
            if isinstance(fusion_partners.gene_3prime_element, GeneElement)
            else None,
            ce,
        )


class GenieTranslator(Translator):
    """Initialize GenieTranslator class"""

    async def translate(
        self,
        genie: Genie,
        coordinate_type: CoordinateType,
        rb: Assembly,
    ) -> AssayedFusion:
        """Parse GENIE output to create AssayedFusion object

        :param genie: Output from GENIE dataset
        :param coordinate_type: If the coordinate is inter-residue or residue
        :param rb: The reference build used to call the fusion
        :return: An AssayedFusion object, if construction is successful
        """
        fusion_partners = self._process_gene_symbols(
            genie.site1_hugo, genie.site2_hugo, Caller.GENIE
        )

        if not self._are_fusion_partners_different(
            fusion_partners.gene_5prime, fusion_partners.gene_3prime
        ):
            return None

        tr_5prime = await self.fusor.transcript_segment_element(
            tx_to_genomic_coords=False,
            genomic_ac=self._get_genomic_ac(genie.site1_chrom, rb),
            seg_end_genomic=genie.site1_pos,
            gene=fusion_partners.gene_5prime,
            coordinate_type=coordinate_type,
            starting_assembly=rb,
        )
        tr_5prime = tr_5prime[0]

        tr_3prime = await self.fusor.transcript_segment_element(
            tx_to_genomic_coords=False,
            genomic_ac=self._get_genomic_ac(genie.site2_chrom, rb),
            seg_start_genomic=genie.site2_pos,
            gene=fusion_partners.gene_3prime,
            coordinate_type=coordinate_type,
            starting_assembly=rb,
        )
        tr_3prime = tr_3prime[0]

        ce = self._get_causative_event(
            genie.site1_chrom,
            genie.site2_chrom,
            genie.annot,
        )
        rf = bool(genie.reading_frame == "in frame")
        return self._format_fusion(
            AssayedFusion,
            fusion_partners.gene_5prime_element,
            fusion_partners.gene_3prime_element,
            tr_5prime
            if isinstance(fusion_partners.gene_5prime_element, GeneElement)
            else None,
            tr_3prime
            if isinstance(fusion_partners.gene_3prime_element, GeneElement)
            else None,
            ce,
            rf,
        )

    ######### Knowledgebase -> FUSOR CategoricalFusion object #############


class CIVICTranslator(Translator):
    """Initialize CIVICTranslator"""

    class Direction(Enum):
        """Define CIViC-specific enum for transcript direction"""

        POSITIVE = "POSITIVE"
        NEGATIVE = "NEGATIVE"

    def _get_breakpoint(
        self, coordinate_data: ExonCoordinate, is_5prime: bool = True
    ) -> int:
        """Extract correct breakpoint for downstream processing

        :param coordinate_data: An ExonCoordinate object
        :param is_5prime: A boolean indicating if 5'partner is being processed
        :return: The modified genomic breakpoint, taking strand, offset, and
            offset direction into account
        """
        if is_5prime:
            coord_to_use = (
                coordinate_data.stop
                if coordinate_data.strand == self.Direction.POSITIVE.value
                else coordinate_data.start
            )
        else:
            coord_to_use = (
                coordinate_data.start
                if coordinate_data.strand == self.Direction.POSITIVE.value
                else coordinate_data.stop
            )
        if coordinate_data.exon_offset_direction == self.Direction.POSITIVE.value:
            return coord_to_use + coordinate_data.exon_offset
        if coordinate_data.exon_offset_direction == self.Direction.NEGATIVE.value:
            return coord_to_use - coordinate_data.exon_offset
        return coord_to_use  # Return current position if exon_offset is 0

    def _valid_exon_coords(self, coord: ExonCoordinate | None) -> bool:
        """Validate exon coordinates

        :param coord: A ExonCoordinate object or None
        :return ``True`` If a start or stop coordinate is associated with the 5'
            end exon or 3` start exon, or ``False``. We cannot peform accurate
            translation with only an Ensembl transcript accession and exon number
        """
        return not (
            isinstance(coord, ExonCoordinate)
            and coord.exon
            and not (coord.start and coord.stop)
        )

    async def translate(self, civic: CIVIC) -> CategoricalFusion | None:
        """Convert CIViC record to Categorical Fusion

        :param civic A CIVIC object
        :return A CategoricalFusion object, if construction is successful
        """
        if not (
            self._valid_exon_coords(civic.five_prime_end_exon_coords)
            and self._valid_exon_coords(civic.three_prime_start_exon_coords)
        ):
            return None
        fusion_partners = civic.vicc_compliant_name
        if fusion_partners.startswith("v::"):
            gene_5prime = "v"
            gene_3prime = self._process_vicc_nomenclature(
                fusion_partners.split("::")[1]
            )
        elif fusion_partners.endswith("::v"):
            gene_5prime = self._process_vicc_nomenclature(
                fusion_partners.split("::")[0]
            )
            gene_3prime = "v"
        else:
            gene_5prime = self._process_vicc_nomenclature(
                fusion_partners.split("::")[0]
            )
            gene_3prime = self._process_vicc_nomenclature(
                fusion_partners.split("::")[1]
            )

        fusion_partners = self._process_gene_symbols(
            gene_5prime, gene_3prime, KnowledgebaseList.CIVIC
        )
        if not self._are_fusion_partners_different(
            fusion_partners.gene_5prime, fusion_partners.gene_3prime
        ):
            return None

        tr_5prime = None
        if (
            isinstance(civic.five_prime_end_exon_coords, ExonCoordinate)
            and civic.five_prime_end_exon_coords.chromosome
        ):  # Process for cases where exon data is available for 5' transcript
            rb = (
                Assembly.GRCH37.value
                if civic.five_prime_end_exon_coords.reference_build == "GRCH37"
                else Assembly.GRCH38.value
            )
            tr_5prime = await self.fusor.transcript_segment_element(
                tx_to_genomic_coords=False,
                genomic_ac=self._get_genomic_ac(
                    civic.five_prime_end_exon_coords.chromosome, rb
                ),
                seg_end_genomic=self._get_breakpoint(
                    civic.five_prime_end_exon_coords, True
                ),
                gene=fusion_partners.gene_5prime,
                coordinate_type=CoordinateType.RESIDUE,
                starting_assembly=rb,
            )
            tr_5prime = tr_5prime[0]

        tr_3prime = None
        if (
            isinstance(civic.three_prime_start_exon_coords, ExonCoordinate)
            and civic.three_prime_start_exon_coords.chromosome
        ):  # Process for case where exon data is available for 3' transcript
            rb = (
                Assembly.GRCH37.value
                if civic.three_prime_start_exon_coords.reference_build == "GRCH37"
                else Assembly.GRCH38.value
            )
            tr_3prime = await self.fusor.transcript_segment_element(
                tx_to_genomic_coords=False,
                genomic_ac=self._get_genomic_ac(
                    civic.three_prime_start_exon_coords.chromosome, rb
                ),
                seg_start_genomic=self._get_breakpoint(
                    civic.three_prime_start_exon_coords, False
                ),
                gene=fusion_partners.gene_3prime,
                coordinate_type=CoordinateType.RESIDUE,
                starting_assembly=rb,
            )
            tr_3prime = tr_3prime[0]

        return self._format_fusion(
            CategoricalFusion,
            fusion_partners.gene_5prime_element,
            fusion_partners.gene_3prime_element,
            tr_5prime if isinstance(tr_5prime, TranscriptSegmentElement) else None,
            tr_3prime if isinstance(tr_3prime, TranscriptSegmentElement) else None,
            molecular_profiles=civic.molecular_profiles,
        )


class MOATranslator(Translator):
    """Initialize MOATranslator"""

    def translate(self, moa_assertion: dict) -> CategoricalFusion | None:
        """Convert a MOA assertion to a CategoricalFusion object

        :param moa_assertion: A dictionary representing a MOA assertion. To note, MOA fusions
            do not report genomic breakpoints. Currently, we only support fusions
            where both partners are listed, as we cannot definitively determine for
            cases where one gene symbol is provided if it describes the 5' or 3'
            partner.
        :return: A CategoricalFusion object or None if both gene partners are not
            provided.
        """
        bm = None
        for biomarker in moa_assertion["proposition"]["biomarkers"]:
            if (
                "::" in biomarker["name"]
            ):  # Extract CategoricalVariant describing fusion
                bm = biomarker["name"]
                break
        moa_partners = bm.split("::")
        gene_5prime = moa_partners[0]
        gene_3prime = moa_partners[1]
        fusion_partners = self._process_gene_symbols(
            gene_5prime, gene_3prime, KnowledgebaseList.MOA
        )
        return self._format_fusion(
            CategoricalFusion,
            fusion_partners.gene_5prime_element,
            fusion_partners.gene_3prime_element,
            moa_assertion=moa_assertion,
        )
